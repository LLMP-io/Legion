import asyncio
import sys
import time
from typing import Any, Dict, List, Optional, Type, Union

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from legion import agent
from legion.agents.base import Agent
from legion.errors import LegionError
from legion.groups.chain import Chain
from legion.interface.schemas import Message, ModelResponse, Role, SystemPrompt
from legion.monitoring.events.chain import (
    ChainBottleneckEvent,
    ChainCompletionEvent,
    ChainErrorEvent,
    ChainStartEvent,
    ChainStateChangeEvent,
    ChainStepEvent,
    ChainTransformEvent,
)

load_dotenv()

# Test Models
class TextInput(BaseModel):
    text: str
    max_length: Optional[int] = None

class TextStats(BaseModel):
    word_count: int
    char_count: int
    avg_word_length: float

class ProcessedText(BaseModel):
    original: str
    processed: str
    stats: TextStats

# Test Agents
@agent(model="gpt-4o-mini")
class TextCleaner:
    """Clean and normalize text input."""

    def process_message(self, message: str) -> ModelResponse:
        if self.debug:
            print(f"Cleaning message: {message}")
        return ModelResponse(
            content=message.strip().lower(),
            raw_response={},
            usage=None,
            tool_calls=None,
            role=Role.USER
        )

@agent(model="gpt-4o-mini")
class TextAnalyzer:
    """Analyze text and compute statistics."""

    async def process_message(self, message: str) -> ModelResponse:
        if self.debug:
            print(f"Analyzing message: {message}")
        total_chars = len(message)
        return ModelResponse(
            content=f"Analysis: '{message}' has {total_chars} total characters",
            raw_response={},
            usage=None,
            tool_calls=None,
            role=Role.USER
        )

@agent(model="gpt-4o-mini")
class TextFormatter:
    """Format text analysis results."""

    async def process_message(self, analysis: str) -> ModelResponse:
        if self.debug:
            print(f"Formatting analysis: {analysis}")
        return ModelResponse(
            content=f"Formatted results: {analysis}",
            raw_response={},
            usage=None,
            tool_calls=None,
            role=Role.USER
        )

@agent(model="gpt-4o-mini")
class ErrorAgent:
    """Agent that raises an error."""

    async def _aprocess(self, message: Union[str, Dict[str, Any], Message], response_schema=None, verbose=False) -> ModelResponse:
        """Process message and raise error"""
        # Create a ModelResponse object before raising the error
        response = ModelResponse(
            content="Error will be raised",
            raw_response={},
            usage=None,
            tool_calls=None
        )
        # Add response to memory
        self.memory.add_message(Message(
            role=Role.ASSISTANT,
            content=response.content
        ))
        # Raise the error
        raise ValueError("Simulated error in ErrorAgent")

# Test Fixtures
@pytest.fixture
def basic_chain():
    return Chain(
        name="basic_chain",
        members=[
            TextCleaner(),
            TextAnalyzer(),
            TextFormatter()
        ]
    )

@pytest.fixture
def nested_chain(basic_chain):
    return Chain(
        name="nested_chain",
        members=[
            TextCleaner(),
            basic_chain,
            TextFormatter()
        ]
    )

@pytest.fixture
def error_chain():
    """Create a chain that raises an error"""
    class ErrorAgent(Agent):
        def __init__(self):
            super().__init__(name="error_agent", model="openai:gpt-3.5-turbo")

        async def _aprocess(
            self,
            message: str,
            response_schema: Optional[Type[BaseModel]] = None,
            dynamic_values: Optional[Dict[str, str]] = None,
            injected_parameters: Optional[List[Dict[str, Any]]] = None,
            verbose: bool = False
        ) -> str:
            raise ValueError("Simulated error in processing")

    class DummyAgent(Agent):
        def __init__(self):
            super().__init__(name="dummy_agent", model="openai:gpt-3.5-turbo")

        async def _aprocess(
            self,
            message: str,
            response_schema: Optional[Type[BaseModel]] = None,
            dynamic_values: Optional[Dict[str, str]] = None,
            injected_parameters: Optional[List[Dict[str, Any]]] = None,
            verbose: bool = False
        ) -> str:
            return message

    return Chain(
        name="error_chain",
        members=[ErrorAgent(), DummyAgent()]
    )

# Basic Chain Tests
def test_chain_initialization():
    """Test chain initialization and validation"""
    # Valid initialization
    chain = Chain(
        name="test_chain",
        members=[TextCleaner(), TextAnalyzer()]
    )
    assert chain.name == "test_chain"
    assert len(chain.members) == 2
    assert chain.metadata.group_type == "chain"

    # Test initialization with single member (should fail)
    with pytest.raises(ValueError) as exc_info:
        Chain(name="invalid_chain", members=[TextCleaner()])
    assert "Chain must have at least 2 members" in str(exc_info.value)

    # Test initialization with empty list
    with pytest.raises(ValueError):
        Chain(name="empty_chain", members=[])

def test_chain_member_setup(basic_chain):
    """Test chain member configuration"""
    members = list(basic_chain.members.values())

    # Check first member
    assert isinstance(members[0].system_prompt, SystemPrompt)
    assert "first" in members[0].system_prompt.render().lower()

    # Check middle member
    assert "position 2" in members[1].system_prompt.render().lower()

    # Check last member
    assert "last" in members[2].system_prompt.render().lower()

def test_chain_processing(basic_chain):
    """Test basic chain processing"""
    result = asyncio.run(basic_chain.process("Test Message"))
    assert isinstance(result, Message)
    assert len(result.content) > 0  # Should have some content

@pytest.mark.asyncio
async def test_chain_async_processing(basic_chain):
    """Test asynchronous chain processing"""
    result = await basic_chain.process("Test Message")
    assert isinstance(result, Message)
    assert len(result.content) > 0  # Should have some content

def test_chain_with_schema(basic_chain):
    """Test chain processing with output schema"""
    class OutputSchema(BaseModel):
        result: str

    result = asyncio.run(basic_chain.process(
        "Test Message",
        response_schema=OutputSchema
    ))
    assert isinstance(result, Message)
    assert len(result.content) > 0  # Should have some content

# Nested Chain Tests
def test_nested_chain_initialization(nested_chain):
    """Test nested chain initialization"""
    assert nested_chain.name == "nested_chain"
    assert len(nested_chain.members) == 3
    assert isinstance(list(nested_chain.members.values())[1], Chain)

@pytest.mark.asyncio
async def test_nested_chain_processing(nested_chain):
    """Test nested chain processing"""
    result = await nested_chain.process("Test Message")
    assert isinstance(result, Message)
    assert len(result.content) > 0  # Should have some content

def test_nested_chain_metadata(nested_chain):
    """Test nested chain metadata"""
    nested = list(nested_chain.members.values())[1]
    assert nested.metadata.group_type == "chain_member"
    assert nested.metadata.position == "position 2"
    assert nested.metadata.depth > 0

def test_circular_reference():
    """Test prevention of circular references"""
    # Create initial chains
    chain1 = Chain(name="chain1", members=[TextCleaner(), TextAnalyzer()])
    chain2 = Chain(name="chain2", members=[TextFormatter(), chain1])

    # Attempt to create circular reference by adding chain2 to chain1
    with pytest.raises(LegionError) as exc_info:
        chain1.add_member("circular", chain2)
    assert "circular reference" in str(exc_info.value).lower()

# Error Handling Tests
def test_chain_error_handling(error_chain):
    """Test error handling in chain processing"""
    # Process message and expect error
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(error_chain.process("Test message"))
    assert "Simulated error in processing" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chain_async_error_handling(error_chain):
    """Test error handling in async chain processing"""
    # Process message and expect error
    with pytest.raises(ValueError) as exc_info:
        await error_chain.process("Test message")
    assert "Simulated error in processing" in str(exc_info.value)

# Member Management Tests
def test_add_member(basic_chain):
    """Test adding new members to chain"""
    new_agent = TextFormatter()
    basic_chain.add_member("formatter2", new_agent)

    assert "formatter2" in basic_chain.members
    assert len(basic_chain.members) == 4
    assert list(basic_chain.members.values())[-1] == new_agent

def test_member_validation():
    """Test member type validation"""
    # Test with invalid member type
    with pytest.raises(TypeError) as exc_info:
        Chain(name="invalid", members=[TextCleaner(), 123])  # Use a number instead of string
    assert "Invalid member type" in str(exc_info.value)

# Debug and Verbose Mode Tests
def test_debug_mode(capsys):
    """Test chain execution in debug mode"""
    chain = Chain(
        name="debug_chain",
        members=[
            TextCleaner(debug=True),
            TextAnalyzer(debug=True)
        ],
        debug=True,
        verbose=True  # Enable verbose mode to see debug output
    )

    asyncio.run(chain.process("Test message"))
    captured = capsys.readouterr()

    # Check debug output
    assert "test message" in captured.out.lower()  # Input message
    assert "textcleaner" in captured.out.lower()  # First member
    assert "textanalyzer" in captured.out.lower()  # Second member

def test_verbose_mode(capsys):
    """Test chain execution in verbose mode"""
    chain = Chain(
        name="verbose_chain",
        members=[TextCleaner(), TextAnalyzer()],
        verbose=True
    )

    asyncio.run(chain.process("Test Message"))
    captured = capsys.readouterr()
    assert "chain structure" in captured.out.lower()
    assert "step" in captured.out.lower()

# Edge Cases
def test_long_chain():
    """Test chain with many members"""
    members = [TextCleaner() for _ in range(10)]
    members.append(TextFormatter())

    chain = Chain(name="long_chain", members=members)
    result = asyncio.run(chain.process("Test Message"))
    assert isinstance(result, Message)
    assert len(result.content) > 0  # Should have some content

def test_chain_with_different_message_types():
    """Test chain with different message input types"""
    chain = Chain(
        name="type_chain",
        members=[TextCleaner(), TextAnalyzer()]
    )

    # Test with string
    result1 = asyncio.run(chain.process("Test Message"))
    assert isinstance(result1, Message)
    assert len(result1.content) > 0

    # Test with Message object
    msg = Message(role=Role.USER, content="Test Message")
    result2 = asyncio.run(chain.process(msg))
    assert isinstance(result2, Message)
    assert len(result2.content) > 0

def test_chain_inheritance():
    """Test chain inheritance and customization"""
    class CustomChain(Chain):
        async def process(self, message, **kwargs):
            # Add custom preprocessing
            if isinstance(message, str):
                message = message.upper()
            return await super().process(message, **kwargs)

    chain = CustomChain(
        name="custom_chain",
        members=[TextCleaner(), TextAnalyzer()]
    )

    result = asyncio.run(chain.process("test message"))
    assert isinstance(result, Message)
    assert len(result.content) > 0

@pytest.mark.asyncio
async def test_chain_concurrent_processing():
    """Test concurrent processing of multiple chains"""
    chains = [
        Chain(name=f"chain_{i}", members=[TextCleaner(), TextAnalyzer()])
        for i in range(3)
    ]

    tasks = [chain.process("Test Message") for chain in chains]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert all(isinstance(r, Message) for r in results)
    assert all(len(r.content) > 0 for r in results)

def test_chain_state_isolation():
    """Test that chains maintain isolated state"""
    chain1 = Chain(name="chain1", members=[TextCleaner(), TextAnalyzer()])
    chain2 = Chain(name="chain2", members=[TextCleaner(), TextAnalyzer()])

    # Process messages through both chains
    result1 = asyncio.run(chain1.process("Message 1"))
    result2 = asyncio.run(chain2.process("Message 2"))

    # Verify results are different
    assert result1.content != result2.content

def test_chain_event_emission():
    """Test chain event emission during processing"""
    chain = Chain(
        name="test_chain",
        members=[TextCleaner(), TextAnalyzer()]
    )

    # Create event handler
    events = []
    def handler(event):
        events.append(event)

    chain.add_event_handler(handler)

    # Process message
    asyncio.run(chain.process("Test Message"))

    # Verify events
    assert len(events) >= 6  # Start, 2x Step, 2x Transform, Completion

    # Check event sequence
    assert isinstance(events[0], ChainStartEvent)
    assert events[0].metadata["member_count"] == 2

    assert isinstance(events[1], ChainStepEvent)
    assert events[1].metadata["step_index"] == 0
    assert not events[1].metadata["is_final_step"]

    assert isinstance(events[2], ChainTransformEvent)
    assert events[2].metadata["step_index"] == 0
    assert events[2].metadata["transformation_time_ms"] > 0

    assert isinstance(events[3], ChainStepEvent)
    assert events[3].metadata["step_index"] == 1
    assert events[3].metadata["is_final_step"]

    assert isinstance(events[4], ChainTransformEvent)
    assert events[4].metadata["step_index"] == 1
    assert events[4].metadata["transformation_time_ms"] > 0

    assert isinstance(events[-1], ChainCompletionEvent)
    assert events[-1].metadata["total_time_ms"] > 0
    assert len(events[-1].metadata["step_times"]) == 2

@pytest.mark.asyncio
async def test_chain_async_event_emission():
    """Test chain event emission during async processing"""
    chain = Chain(
        name="test_chain",
        members=[TextCleaner(), TextAnalyzer()]
    )

    # Create event handler
    events = []
    def handler(event):
        events.append(event)

    chain.add_event_handler(handler)

    # Process message
    await chain.process("Test Message")

    # Verify events
    assert len(events) >= 6  # Start, 2x Step, 2x Transform, Completion

    # Check event sequence
    assert isinstance(events[0], ChainStartEvent)
    assert events[0].metadata["member_count"] == 2

    assert isinstance(events[1], ChainStepEvent)
    assert events[1].metadata["step_index"] == 0
    assert not events[1].metadata["is_final_step"]

    assert isinstance(events[2], ChainTransformEvent)
    assert events[2].metadata["step_index"] == 0
    assert events[2].metadata["transformation_time_ms"] > 0

    assert isinstance(events[3], ChainStepEvent)
    assert events[3].metadata["step_index"] == 1
    assert events[3].metadata["is_final_step"]

    assert isinstance(events[4], ChainTransformEvent)
    assert events[4].metadata["step_index"] == 1
    assert events[4].metadata["transformation_time_ms"] > 0

    assert isinstance(events[-1], ChainCompletionEvent)
    assert events[-1].metadata["total_time_ms"] > 0
    assert len(events[-1].metadata["step_times"]) == 2

def test_chain_error_event():
    """Test error event emission"""
    class ErrorAgent(Agent):
        def __init__(self):
            super().__init__(name="error_agent", model="gpt-4o-mini")

        async def _aprocess(
            self,
            message: str,
            response_schema: Optional[Type[BaseModel]] = None,
            dynamic_values: Optional[Dict[str, str]] = None,
            injected_parameters: Optional[List[Dict[str, Any]]] = None,
            verbose: bool = False
        ) -> str:
            raise ValueError("Test error")

    chain = Chain(
        name="error_chain",
        members=[TextCleaner(), ErrorAgent()]
    )

    # Create event handler
    events = []
    def handler(event):
        events.append(event)

    chain.add_event_handler(handler)

    # Process message and expect error
    with pytest.raises(ValueError):
        asyncio.run(chain.process("Test Message"))

    # Verify error event
    error_events = [e for e in events if isinstance(e, ChainErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].metadata["error_type"] == "ValueError"
    assert error_events[0].metadata["error_message"] == "Test error"
    assert error_events[0].metadata["step_index"] == 1

def test_chain_bottleneck_detection(monkeypatch):
    """Test bottleneck detection and event emission"""
    # Mock time.time to control timing
    start_time = 0.0
    def mock_time():
        nonlocal start_time
        start_time += 2.0  # Each call advances by 2 seconds
        return start_time

    monkeypatch.setattr(time, "time", mock_time)

    class SlowAgent(Agent):
        def __init__(self):
            super().__init__(name="slow_agent", model="gpt-4o-mini")

        async def _aprocess(
            self,
            message: str,
            response_schema: Optional[Type[BaseModel]] = None,
            dynamic_values: Optional[Dict[str, str]] = None,
            injected_parameters: Optional[List[Dict[str, Any]]] = None,
            verbose: bool = False
        ) -> str:
            content = message.content if isinstance(message, Message) else str(message)
            return ModelResponse(
                content=content,
                raw_response={},
                usage=None,
                tool_calls=None,
                role=Role.USER
            )

    chain = Chain(
        name="slow_chain",
        members=[TextCleaner(), SlowAgent()]
    )

    # Lower bottleneck threshold for testing
    chain._bottleneck_threshold_ms = 100.0

    # Create event handler
    events = []
    def handler(event):
        events.append(event)

    chain.add_event_handler(handler)

    # Process messages to trigger bottleneck
    asyncio.run(chain.process("Test Message 1"))
    asyncio.run(chain.process("Test Message 2"))

    # Verify bottleneck events
    bottleneck_events = [e for e in events if isinstance(e, ChainBottleneckEvent)]
    assert len(bottleneck_events) > 0
    assert bottleneck_events[0].metadata["step_index"] == 1
    assert bottleneck_events[0].metadata["processing_time_ms"] > 1500  # Above threshold
    assert bottleneck_events[0].metadata["processing_time_ms"] > bottleneck_events[0].metadata["threshold_ms"]

def test_chain_state_change_event():
    """Test state change event emission"""
    chain = Chain(
        name="test_chain",
        members=[TextCleaner(), TextAnalyzer()]
    )

    # Create event handler
    events = []
    def handler(event):
        events.append(event)

    chain.add_event_handler(handler)

    # Add new member
    chain.add_member("step_3", TextFormatter())

    # Verify state change event
    state_events = [e for e in events if isinstance(e, ChainStateChangeEvent)]
    assert len(state_events) == 1
    assert state_events[0].metadata["change_type"] == "member_added"
    assert state_events[0].metadata["old_state"]["member_count"] == 2
    assert state_events[0].metadata["new_state"]["member_count"] == 3

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
