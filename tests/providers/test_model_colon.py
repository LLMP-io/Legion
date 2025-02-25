"""Test Model Colon Parsing

This test verifies the fix for models with colons in their names,
such as 'codellama:70b'.
"""

import os
import pytest
from typing import Annotated, Tuple

from dotenv import load_dotenv
from pydantic import Field

from legion import agent, tool
from legion.interface.schemas import ModelResponse

# Load environment variables
load_dotenv()

# Check for API keys
MOCK_MODE = not bool(os.getenv("OPENAI_API_KEY"))


# Test cases for model parsing
MODEL_TEST_CASES = [
    ("openai:gpt-4", "openai", "gpt-4"),
    ("anthropic:claude-3-opus", "anthropic", "claude-3-opus"),
    ("ollama:codellama:70b", "ollama", "codellama:70b"),
    ("groq:mixtral-8x7b", "groq", "mixtral-8x7b"),
]


# Define a mock agent that simulates the behavior without making actual API calls
@agent(
    model="openai:gpt-4",  # Using OpenAI since it's fully implemented
    temperature=0.7
)
class TestAgent:
    """I am an agent that tests model names with colons."""

    @tool
    def get_model_info(self) -> str:
        """Return information about the model being used."""
        if MOCK_MODE:
            return "Mock mode: Using openai:gpt-4"
        else:
            return "Using openai:gpt-4"


@pytest.mark.parametrize("model_str,expected_provider,expected_model", MODEL_TEST_CASES)
def test_model_parsing(model_str: str, expected_provider: str, expected_model: str):
    """Test the model parsing function directly."""
    from legion.agents.base import Agent
    
    provider, model = Agent._parse_model_string(model_str)
    
    assert provider == expected_provider
    assert model == expected_model


@pytest.mark.asyncio
async def test_agent_with_colon_model():
    """Test that an agent can be created with a model name containing colons."""
    # This test just verifies that the agent can be instantiated without errors
    agent = TestAgent()
    assert agent is not None
    
    # If not in mock mode, test actual processing
    if not MOCK_MODE:
        response = await agent.aprocess("What model are you using?")
        assert isinstance(response, ModelResponse)
        assert response.content is not None 