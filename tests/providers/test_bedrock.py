import json
import os
from typing import List

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from legion.errors import ProviderError
from legion.interface.schemas import Message, ModelResponse, ProviderConfig, Role
from legion.interface.tools import BaseTool
from legion.providers.bedrock import BedrockFactory, BedrockProvider

# Load environment variables
load_dotenv()

class TestSchema(BaseModel):
    name: str
    age: int
    hobbies: List[str]

class SimpleToolParams(BaseModel):
    message: str

class SimpleTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="simple_tool",
            description="A simple test tool",
            parameters=SimpleToolParams
        )

    def run(self, message: str) -> str:
        """Implement the sync run method"""
        return f"Tool response: {message}"

    async def arun(self, message: str) -> str:
        """Implement the async run method"""
        return self.run(message)

@pytest.fixture
def provider():
    config = ProviderConfig(
        api_key=os.getenv("AWS_ACCESS_KEY_ID"),
        api_secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region=os.getenv("AWS_REGION", "us-east-1")
    )
    return BedrockProvider(config=config)

@pytest.fixture
def factory():
    return BedrockFactory()

def test_provider_initialization(provider):
    assert isinstance(provider, BedrockProvider)
    assert provider._client is not None

def test_factory_creation(factory):
    config = ProviderConfig(
        api_key=os.getenv("AWS_ACCESS_KEY_ID"),
        api_secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region=os.getenv("AWS_REGION", "us-east-1")
    )
    provider = factory.create_provider(config=config)
    assert isinstance(provider, BedrockProvider)

@pytest.mark.asyncio
async def test_basic_completion(provider):
    messages = [
        Message(role=Role.USER, content="Say hello")
    ]

    response = await provider.acomplete(
        messages=messages,
        model="us.amazon.nova-lite-v1:0",
        temperature=0
    )

    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_tool_completion(provider):
    tool = SimpleTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools when appropriate."),
        Message(role=Role.USER, content="Please use the simple_tool with the message 'hello world'")
    ]

    response = await provider.acomplete(
        messages=messages,
        model="us.amazon.nova-lite-v1:0",
        tools=[tool],
        temperature=0
    )

    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "simple_tool"
    assert "hello world" in response.tool_calls[0]["function"]["arguments"].lower()

@pytest.mark.asyncio
async def test_json_completion(provider):
    messages = [
        Message(
            role=Role.USER,
            content="Create a JSON object for a person with these exact details: name is John, age is 25, and hobbies are reading and gaming. Make sure to follow the schema exactly."
        )
    ]

    response = await provider.acomplete(
        messages=messages,
        model="us.amazon.nova-lite-v1:0",
        response_schema=TestSchema,
        temperature=0
    )

    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

    # Verify that the response can be parsed as JSON
    data = json.loads(response.content)
    schema = TestSchema(**data)
    assert schema.name == "John"
    assert schema.age == 25
    assert "reading" in schema.hobbies
    assert "gaming" in schema.hobbies

@pytest.mark.asyncio
async def test_tool_and_json_completion(provider):
    tool = SimpleTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools when appropriate."),
        Message(
            role=Role.USER,
            content="First use the simple_tool with the message 'hello world', then create a JSON response about a person named Alice who is 30 and likes dancing and singing."
        )
    ]

    response = await provider.acomplete(
        messages=messages,
        model="us.amazon.nova-lite-v1:0",
        tools=[tool],
        response_schema=TestSchema,
        temperature=0
    )

    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0

    # Verify that the response can be parsed as JSON
    data = json.loads(response.content)
    schema = TestSchema(**data)
    assert schema.name == "Alice"
    assert schema.age == 30
    assert "dancing" in schema.hobbies
    assert "singing" in schema.hobbies

@pytest.mark.asyncio
async def test_invalid_credentials(provider):
    messages = [Message(role=Role.USER, content="Say hello")]

    # Create a new provider with invalid credentials
    invalid_config = ProviderConfig(
        api_key="invalid_key",
        api_secret="invalid_secret",
        region="us-east-1"
    )
    invalid_provider = BedrockProvider(config=invalid_config)

    with pytest.raises(ProviderError):
        await invalid_provider.acomplete(
            messages=messages,
            model="us.amazon.nova-lite-v1:0",
            temperature=0
        )

@pytest.mark.asyncio
async def test_invalid_model(provider):
    messages = [Message(role=Role.USER, content="Say hello")]
    with pytest.raises(ProviderError):
        await provider.acomplete(
            messages=messages,
            model="invalid-model",
            temperature=0
        )

@pytest.mark.asyncio
async def test_system_message_handling(provider):
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that always responds with 'Hello'"),
        Message(role=Role.USER, content="Say something")
    ]

    response = await provider.acomplete(
        messages=messages,
        model="us.amazon.nova-lite-v1:0",
        temperature=0
    )

    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert "Hello" in response.content
