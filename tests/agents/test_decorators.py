import sys
from typing import Annotated

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from legion import agent, tool, system_prompt
from legion.agents.base import Agent
from legion.interface.schemas import ModelResponse, SystemPrompt, SystemPromptSection

# Load environment variables
load_dotenv()

# Test schemas
class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

# Define a reusable test tool
@tool
def simple_tool(message: Annotated[str, Field(description="A message to process")]) -> str:
    """A simple test tool"""
    return f"Tool response: {message}"

def test_basic_decorator():
    """Test basic agent decorator without tools"""

    @agent(model="gpt-4o-mini")
    class SimpleAgent:
        """I am a simple test agent."""

        pass

    # Create instance
    simple_agent = SimpleAgent()

    # Verify initialization
    assert isinstance(simple_agent, Agent)
    assert simple_agent.name == "SimpleAgent"
    assert simple_agent.model == "gpt-4o-mini"
    assert simple_agent.temperature == 0.7  # default value
    assert simple_agent.system_prompt.render() == "I am a simple test agent."
    assert len(simple_agent.tools) == 0

def test_decorator_with_params():
    """Test agent decorator with custom parameters"""

    @agent(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=100,
        system_prompt="Custom system prompt"
    )
    class CustomAgent:
        pass

    # Create instance
    custom_agent = CustomAgent()

    # Verify initialization
    assert custom_agent.name == "CustomAgent"  # Class name takes precedence
    assert custom_agent.model == "gpt-4o-mini"
    assert custom_agent.temperature == 0.5
    assert custom_agent.max_tokens == 100
    assert custom_agent.system_prompt.render() == "Custom system prompt"

def test_decorator_with_tools():
    """Test agent decorator with tool integration"""

    @agent(
        model="gpt-4o-mini",
        tools=[simple_tool]  # Bind external tool
    )
    class ToolAgent:
        """I am an agent that uses tools."""

        # Internal tool
        @tool
        def internal_tool(
            self,
            message: Annotated[str, Field(description="A message to process")]
        ) -> str:
            """An internal test tool"""
            return f"Internal tool: {message}"

    # Create instance
    tool_agent = ToolAgent()

    # Verify tool integration
    assert len(tool_agent.tools) == 2  # Both external and internal tools
    assert any(t.name == "simple_tool" for t in tool_agent.tools)
    assert any(t.name == "internal_tool" for t in tool_agent.tools)

def test_decorator_with_system_prompt_sections():
    """Test agent decorator with dynamic SystemPrompt sections"""

    def get_test_value() -> str:
        return "dynamic test value"

    # Create a dynamic system prompt
    test_prompt = SystemPrompt(
        sections=[
            SystemPromptSection(
                content="Static section content",
                is_dynamic=False
            ),
            SystemPromptSection(
                content="{dynamic_value}",
                is_dynamic=True,
                section_id="dynamic_section",
                default_value="default value"
            ),
            SystemPromptSection(
                content="{computed}",
                is_dynamic=True,
                section_id="computed_section",
                default_value=get_test_value
            )
        ]
    )

    @agent(
        model="gpt-4o-mini",
        system_prompt=test_prompt
    )
    class SectionAgent:
        # No docstring - using dynamic system prompt
        pass

    # Create instance
    section_agent = SectionAgent()

    # Verify system prompt structure
    assert isinstance(section_agent.system_prompt, SystemPrompt)
    assert len(section_agent.system_prompt.sections) == 3

    # Test default rendering
    rendered = section_agent.system_prompt.render()
    assert "Static section content" in rendered
    assert "dynamic_section: default value" in rendered
    assert "computed_section: dynamic test value" in rendered

    # Test dynamic value rendering
    rendered = section_agent.system_prompt.render({
        "dynamic_section": "custom value",
        "computed_section": "override value"
    })
    assert "Static section content" in rendered
    assert "dynamic_section: custom value" in rendered
    assert "computed_section: override value" in rendered

def test_decorator_inheritance():
    """Test agent decorator with class inheritance"""

    @agent(model="gpt-4o-mini")
    class BaseAgent:
        """I am a base agent."""

        def custom_method(self) -> str:
            return "base method"

    class DerivedAgent(BaseAgent):
        """I am a derived agent."""

        def custom_method(self) -> str:
            return "derived method"

    # Create instances
    base_agent = BaseAgent()
    derived_agent = DerivedAgent()

    # Verify inheritance
    assert isinstance(base_agent, Agent)
    assert isinstance(derived_agent, Agent)
    assert base_agent.custom_method() == "base method"
    assert derived_agent.custom_method() == "derived method"

def test_decorator_with_json_schema():
    """Test agent decorator with JSON schema support"""

    @agent(model="gpt-4o-mini")
    class SchemaAgent:
        """I am an agent that returns structured data."""

        pass

    # Create instance
    schema_agent = SchemaAgent()

    # Test JSON schema completion
    response = schema_agent.process(
        "Give me information about a person named John who is 30 and works as a developer",
        response_schema=PersonInfo
    )
    assert isinstance(response, ModelResponse)
    data = PersonInfo.model_validate_json(response.content)
    assert isinstance(data.name, str)
    assert isinstance(data.age, int)
    assert isinstance(data.occupation, str)

@pytest.mark.asyncio
async def test_decorator_async():
    """Test agent decorator with async processing"""

    @agent(
        model="gpt-4o-mini",
        tools=[simple_tool]
    )
    class AsyncAgent:
        """I am an async agent."""

        pass

    # Create instance
    async_agent = AsyncAgent()

    # Test async completion
    response = await async_agent.aprocess("Use the simple tool to say hello")
    assert isinstance(response, ModelResponse)

def test_decorator_validation():
    """Test agent decorator parameter validation"""
    # Test invalid temperature
    with pytest.raises(ValueError):
        @agent(model="gpt-4-mini", temperature=2.0)
        class InvalidAgent:
            pass

def test_decorator_with_custom_init():
    """Test agent decorator with custom __init__ method"""

    @agent(
        model="gpt-4o-mini",
        tools=[simple_tool]
    )
    class CustomInitAgent:
        """I am an agent with custom initialization."""

        def __init__(self, custom_param: str):
            self.custom_param = custom_param

    # Create instance with custom parameter
    custom_agent = CustomInitAgent("test_param")

    # Verify custom initialization
    assert custom_agent.custom_param == "test_param"
    assert len(custom_agent.tools) == 1
    assert custom_agent.tools[0].name == "simple_tool"

if __name__ == "__main__":
    # Configure pytest arguments
    args = [
        __file__,
        "-v",
        "-p", "no:warnings",
        "--tb=short",
        "--asyncio-mode=auto"
    ]

    # Run tests
    sys.exit(pytest.main(args))
