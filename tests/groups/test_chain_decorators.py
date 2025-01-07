"""Tests for chain decorators"""

import pytest
import asyncio
import os
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("\nError: OpenAI API key not found!")
    print("Please set your API key in the environment:")
    print("    export OPENAI_API_KEY=your_api_key_here")
    exit(1)

from legion.agents.decorators import agent
from legion.blocks.decorators import block
from legion.groups.decorators import chain
from legion.interface.schemas import SystemPrompt, SystemPromptSection

# Test schemas
class InputData(BaseModel):
    text: str
    metadata: Dict[str, Any]

class OutputData(BaseModel):
    summary: str
    word_count: int

# Test blocks and agents
@block(output_schema=InputData)
def preprocess_data(text: str) -> Dict:
    """Preprocess input data"""
    return {
        "text": text.strip(),
        "metadata": {"length": len(text)}
    }

@agent(
    model="openai:gpt-4o-mini",
    system_prompt=SystemPrompt(sections=[
        SystemPromptSection(
            content="You are a test agent that processes data."
        )
    ])
)
class TestAgent:
    """Test agent that processes data"""
    pass

@block(output_schema=OutputData)
def postprocess_data(text: str) -> Dict:
    """Postprocess output data"""
    return {
        "summary": text[:100],
        "word_count": len(text.split())
    }

# Test chain with error
@agent(
    model="openai:gpt-4o-mini",
    system_prompt=SystemPrompt(sections=[
        SystemPromptSection(
            content="You are a test agent that raises errors."
        )
    ])
)
class ErrorAgent:
    """Test agent that raises errors"""
    async def _aprocess(
        self,
        message: str,
        response_schema: Optional[Type[BaseModel]] = None,
        dynamic_values: Optional[Dict[str, str]] = None,
        verbose: bool = False
    ) -> str:
        raise ValueError("Test error")

# Test chains
@chain(name="BasicChain")
class BasicChain:
    """Basic test chain"""
    members = [
        preprocess_data,
        TestAgent,
        postprocess_data
    ]

@chain(name="ErrorChain")
class ErrorChain:
    """Chain that raises an error"""
    members = [
        preprocess_data,
        ErrorAgent,
        postprocess_data
    ]

def test_chain_error_handling():
    """Test that chain properly handles errors"""
    chain = ErrorChain()
    
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(chain.process("test input"))
    assert str(exc_info.value) == "Test error"

async def test_chain_error_handling_async():
    """Test that chain properly handles errors in async mode"""
    chain = ErrorChain()
    
    with pytest.raises(ValueError) as exc_info:
        await chain.process("test input")
    assert str(exc_info.value) == "Test error"

def test_chain_processing():
    """Test basic chain processing"""
    chain = BasicChain()
    result = asyncio.run(chain.process("test input"))
    assert isinstance(result.content, str)

async def test_chain_processing_async():
    """Test basic chain processing in async mode"""
    chain = BasicChain()
    result = await chain.process("test input")
    assert isinstance(result.content, str)

if __name__ == "__main__":
    # Run sync tests
    test_chain_error_handling()
    test_chain_processing()
    
    # Run async tests
    asyncio.run(test_chain_error_handling_async())
    asyncio.run(test_chain_processing_async())