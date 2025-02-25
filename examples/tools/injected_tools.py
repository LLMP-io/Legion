"""Parameter Injection Example

This example demonstrates how to use parameter injection with tools in Legion.
It shows how to use per-message parameter injection and default values.
"""

from typing import Annotated, Any, Dict, List  # noqa: F401

from colorama import Fore, Style, init
from dotenv import load_dotenv
from pydantic import Field

from legion import agent, tool


load_dotenv()

init()  # Initialize colorama


# Define a tool with injectable parameters using the @tool decorator
@tool(
    inject=["api_key", "endpoint", "custom_header"],  # Mark these as injectable
    description="Process a query using an external API with configurable parameters",
    defaults={  # Default values for injectable parameters
        "api_key": "sk_test_default_key",
        "endpoint": "https://api.example.com/v1",
        "custom_header": "default-header"
    }
)
def process_api_query(
    query: Annotated[str, Field(description="The query to process")],
    api_key: Annotated[str, Field(description="API key for the service", default=None)],
    endpoint: Annotated[str, Field(description="API endpoint", default=None)],
    custom_header: Annotated[str, Field(
        description="Custom header for the request",
        default=None
    )]
) -> str:
    """Process a query using an external API with injected credentials and headers."""
    print(f"{Fore.CYAN}[TOOL EXECUTION]{Style.RESET_ALL} process_api_query called with:")
    print(f"{Fore.CYAN}  - query:{Style.RESET_ALL} {query}")
    print(f"{Fore.CYAN}  - endpoint:{Style.RESET_ALL} {endpoint}")
    print(f"{Fore.CYAN}  - api_key:{Style.RESET_ALL} {api_key[:4]}...")
    print(f"{Fore.CYAN}  - custom_header:{Style.RESET_ALL} {custom_header}")

    # In a real application, you would make an actual API call here
    return (
        f"Processed '{query}' using API at {endpoint}\n"
        f"Auth: {api_key[:4]}...\n"
        f"Header: {custom_header}"
    )


@agent(
    model="openai:gpt-4o-mini",
    temperature=0.2,
    tools=[process_api_query],  # Bind the tool
)
class APIAgent:
    """An agent that demonstrates using tools with injected parameters.

    I can process queries using an external API without exposing sensitive credentials.
    Parameters are injected per-message with optional defaults.
    """

    @tool
    def format_api_response(
        self,
        response: Annotated[str, Field(description="API response to format")],
        prefix: Annotated[str, Field(description="Prefix to add")] = "Result: "
    ) -> str:
        """Format an API response with a prefix."""
        print(f"{Fore.CYAN}[TOOL EXECUTION]{Style.RESET_ALL} format_api_response called:")
        print(f"{Fore.CYAN}  - response:{Style.RESET_ALL} {response}")
        print(f"{Fore.CYAN}  - prefix:{Style.RESET_ALL} {prefix}")
        return f"{prefix}{response}"


async def main():
    print(f"{Fore.GREEN}[MAIN]{Style.RESET_ALL} Creating agent instance")
    # Create an instance of our agent
    agent = APIAgent()

    print(f"\n{Fore.GREEN}[MAIN]{Style.RESET_ALL} Running Example 1: Using defaults")
    # Example 1: Using default values
    response = await agent.aprocess(
        "Process the query 'hello world' using our API and format the result nicely."
    )
    print(f"{Fore.YELLOW}Example 1 Response (using default values):{Style.RESET_ALL}")
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    print(f"{Fore.GREEN}[MAIN]{Style.RESET_ALL} Running Example 2: Custom credentials")
    # Example 2: Override API credentials
    response = await agent.aprocess(
        "Process the query 'test message' with production credentials.",
        injected_parameters=[
            {
                "tool": process_api_query,
                "parameters": {
                    "api_key": "sk_prod_key_123",
                    "endpoint": "https://api.prod.example.com/v1"
                }
            }
        ]
    )
    print(f"{Fore.YELLOW}Example 2 Response (with production creds):{Style.RESET_ALL}")
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    print(f"{Fore.GREEN}[MAIN]{Style.RESET_ALL} Running Example 3: Custom header")
    # Example 3: Multiple queries with custom header
    response = await agent.aprocess(
        "Process these two queries with different headers: 'query1' and 'query2'.",
        injected_parameters=[
            {
                "tool": process_api_query,
                "parameters": {
                    "custom_header": "header-for-batch-request"
                }
            }
        ]
    )
    print(f"{Fore.YELLOW}Example 3 Response (with custom header):{Style.RESET_ALL}")
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    print(f"{Fore.GREEN}[MAIN]{Style.RESET_ALL} Running Example 4: Security check")
    # Example 4: Show how the agent can't access injected parameters
    response = await agent.aprocess(
        "What is the API endpoint, key, and custom header we're using?"
    )
    print(f"{Fore.YELLOW}Example 4 Response (security check):{Style.RESET_ALL}")
    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
