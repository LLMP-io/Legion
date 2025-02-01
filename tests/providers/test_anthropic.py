"""Tests for the Anthropic provider implementation"""

import os
from typing import List

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from legion.errors import ProviderError
from legion.interface.schemas import Message, ModelResponse, ProviderConfig, Role
from legion.interface.tools import BaseTool
from legion.providers.anthropic import AnthropicFactory, AnthropicProvider

# Load environment variables
load_dotenv()

class TestResponse(BaseModel):
    message: str
    score: float
    tags: List[str]

class MockToolParams(BaseModel):
    input: str

class MockTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="mock_tool",
            description="A mock tool for testing",
            parameters=MockToolParams
        )

    def run(self, input: str) -> str:
        return f"Mock tool response: {input}"

    async def arun(self, input: str) -> str:
        return self.run(input)

@pytest.fixture
def provider():
    config = ProviderConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY", "test-key")
    )
    return AnthropicProvider(config=config)

@pytest.fixture
def factory():
    return AnthropicFactory()

def test_provider_creation(factory):
    """Test provider creation through factory"""
    config = ProviderConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY", "test-key")
    )
    provider = factory.create_provider(config=config)
    assert isinstance(provider, AnthropicProvider)

def test_provider_initialization(provider):
    """Test provider initialization"""
    assert isinstance(provider, AnthropicProvider)
    assert provider.client is not None

def test_message_formatting(provider):
    """Test message formatting"""
    messages = [
        Message(role=Role.SYSTEM, content="System message"),
        Message(role=Role.USER, content="User message"),
        Message(role=Role.ASSISTANT, content="Assistant message"),
        Message(role=Role.TOOL, content="Tool message", tool_call_id="123", name="tool_name")
    ]
    formatted = provider._format_messages(messages)
    assert isinstance(formatted, list)
    assert len(formatted) == 3  # System message handled separately
    assert all(isinstance(msg, dict) for msg in formatted)

@pytest.mark.asyncio
async def test_basic_completion(provider):
    """Test basic completion"""
    messages = [
        Message(role=Role.USER, content="Say 'Hello, World!'")
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        temperature=0.7
    )
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.usage is not None
    assert response.tool_calls is None

@pytest.mark.asyncio
async def test_tool_completion(provider):
    """Test completion with tool use"""
    tool = MockTool()
    messages = [
        Message(role=Role.USER, content="Use the mock tool with input='test'")
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        tools=[tool],
        temperature=0.7
    )
    assert isinstance(response, ModelResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "mock_tool"

@pytest.mark.asyncio
async def test_json_completion(provider):
    """Test JSON completion"""
    messages = [
        Message(
            role=Role.USER,
            content="Generate a test response with message='Hello', score=0.9, tags=['test']"
        )
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        temperature=0.1,
        response_schema=TestResponse
    )
    assert response.content
    data = TestResponse.model_validate_json(response.content)
    assert isinstance(data.message, str)
    assert isinstance(data.score, float)
    assert isinstance(data.tags, list)

@pytest.mark.asyncio
async def test_tool_and_json_completion(provider):
    """Test combining tool use with JSON response formatting"""
    tool = MockTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools and returns structured data."),
        Message(role=Role.USER, content="Use the mock tool with input='test', then format the response as a test response")
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        tools=[tool],
        temperature=0.7,
        response_schema=TestResponse
    )
    assert response.content
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "mock_tool"

    # Verify JSON response
    data = TestResponse.model_validate_json(response.content)
    assert isinstance(data.message, str)
    assert isinstance(data.score, float)
    assert isinstance(data.tags, list)

@pytest.mark.asyncio
async def test_async_completion(provider):
    """Test async completion"""
    messages = [
        Message(role=Role.USER, content="Say 'Hello, World!'")
    ]
    response = await provider.acomplete(
        messages=messages,
        model="claude-3-haiku-20240307",
        temperature=0.7
    )
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_system_message_handling(provider):
    """Test system message handling"""
    messages = [
        Message(role=Role.SYSTEM, content="You are Claude, a helpful AI assistant created by Anthropic."),
        Message(role=Role.USER, content="Who are you?")
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    assert "Claude" in response.content

@pytest.mark.asyncio
async def test_invalid_model(provider):
    """Test error handling for invalid model"""
    messages = [Message(role=Role.USER, content="Test")]
    with pytest.raises(ProviderError):
        await provider.complete(
            messages=messages,
            model="invalid-model"
        )

@pytest.mark.asyncio
async def test_invalid_api_key():
    """Test error handling for invalid API key"""
    config = ProviderConfig(api_key="invalid_key")
    provider = AnthropicProvider(config=config)
    messages = [Message(role=Role.USER, content="Test")]
    with pytest.raises(ProviderError):
        await provider.complete(
            messages=messages,
            model="claude-3-haiku-20240307"
        )

if __name__ == "__main__":
    pytest.main()
