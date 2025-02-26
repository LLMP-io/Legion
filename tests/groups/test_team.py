import json
import sys
from typing import Annotated, Any, Dict, Union

import pytest
from dotenv import load_dotenv
from pydantic import Field

from legion import agent, tool
from legion.groups.team import Team
from legion.interface.schemas import Message, ModelResponse, Role

# Load environment variables for OpenAI API key
load_dotenv()

# Test Agents
@agent(
    model="gpt-4o-mini",
    temperature=0  # Add temperature=0 for deterministic tests
)
class TeamLeader:
    """Leader agent that can delegate tasks to team members."""

    @tool
    def researcher(
        self,
        query: Annotated[str, Field(description="The information to analyze")]
    ) -> str:
        """Delegate analysis and research tasks to the researcher."""
        return query

    @tool
    def writer(
        self,
        content: Annotated[str, Field(description="The content to format")]
    ) -> str:
        """Delegate writing and formatting tasks to the writer."""
        return content

    async def aprocess(
        self,
        message: Union[str, Dict[str, Any], Message],
        response_schema=None
    ) -> ModelResponse:
        """Process message and delegate to appropriate team member.
        For testing, we always delegate to researcher to ensure consistent behavior.
        """
        # Get message content
        msg_content = message.content if isinstance(message, Message) else str(message)

        # For testing, we want to ensure the tool is always used
        return ModelResponse(
            content="Delegating to researcher",
            tool_calls=[{
                "function": {
                    "name": "researcher",
                    "arguments": json.dumps({"query": msg_content})
                }
            }],
            model="gpt-4o-mini",
            role=Role.ASSISTANT
        )

@agent(model="gpt-4o-mini", temperature=0)
class Researcher:
    """Research agent that analyzes information."""

    async def aprocess(
        self,
        message: Union[str, Dict[str, Any], Message],
        response_schema=None
    ) -> ModelResponse:
        # Get message content
        msg_content = message.content if isinstance(message, Message) else str(message)
        # If content is JSON, try to parse it
        try:
            args = json.loads(msg_content)
            msg_content = args.get("query", msg_content)
        except (json.JSONDecodeError, AttributeError):
            pass
        return ModelResponse(
            content=f"Research findings for: {msg_content}",
            model="gpt-4o-mini",
            role=Role.ASSISTANT
        )

@agent(model="gpt-4o-mini", temperature=0)
class Writer:
    """Writing agent that formats content."""

    async def aprocess(
        self,
        message: Union[str, Dict[str, Any], Message],
        response_schema=None
    ) -> ModelResponse:
        # Get message content
        msg_content = message.content if isinstance(message, Message) else str(message)
        # If content is JSON, try to parse it
        try:
            args = json.loads(msg_content)
            msg_content = args.get("content", msg_content)
        except (json.JSONDecodeError, AttributeError):
            pass
        return ModelResponse(
            content=f"Formatted content: {msg_content}",
            model="gpt-4o-mini",
            role=Role.ASSISTANT
        )

@pytest.fixture
def basic_team():
    """Create a basic team for testing."""
    leader = TeamLeader()
    members = {
        "researcher": Researcher(),
        "writer": Writer()
    }
    return Team(name="TestTeam", leader=leader, members=members)

def test_team_initialization(basic_team):
    """Test basic team initialization."""
    assert basic_team.name == "TestTeam"
    assert isinstance(basic_team.leader, TeamLeader)
    assert len(basic_team.members) == 2
    assert isinstance(basic_team.members["researcher"], Researcher)
    assert isinstance(basic_team.members["writer"], Writer)

def test_team_delegation_recording(basic_team):
    """Test recording of task delegations."""
    basic_team._record_delegation(
        source="leader",
        target="researcher",
        task="analyze data"
    )
    assert len(basic_team._delegation_history) == 1
    delegation = basic_team._delegation_history[0]
    assert delegation["source"] == "leader"
    assert delegation["target"] == "researcher"
    assert delegation["task"] == "analyze data"

def test_team_context_tracking(basic_team):
    """Test context tracking for team members."""
    # Record some context
    basic_team._last_context.update({
        "last_message": "test message",
        "message_type": "input",
        "source": "user"
    })

    # Record a delegation
    basic_team._record_delegation(
        source="leader",
        target="researcher",
        task="analyze test message"
    )

    # Get context for researcher
    context = basic_team._get_context_for_member("researcher")
    assert context["last_message"] == "test message"
    assert context["message_type"] == "input"
    assert context["source"] == "user"
    assert len(context["delegations"]) == 1

@pytest.mark.asyncio
async def test_team_async_processing(basic_team):
    """Test async message processing through the team."""
    message = Message(content="Please analyze this information")
    response = await basic_team.aprocess(message)

    # Verify response
    assert isinstance(response, ModelResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "researcher"
    assert "result" in response.tool_calls[0]
    assert "Research findings" in response.tool_calls[0]["result"]

def test_team_sync_processing_not_supported(basic_team):
    """Test that sync processing raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        basic_team.process(Message(content="test"))

@pytest.mark.asyncio
async def test_team_multiple_delegations():
    """Test team handling multiple delegations."""
    @agent(model="gpt-4o-mini", temperature=0)
    class MultiDelegationLeader:
        """Leader that can delegate to multiple team members."""

        @tool
        def researcher(
            self,
            query: Annotated[str, Field(description="The information to analyze")]
        ) -> str:
            """Delegate analysis tasks to researcher."""
            return query

        @tool
        def writer(
            self,
            content: Annotated[str, Field(description="The content to format")]
        ) -> str:
            """Delegate writing tasks to writer."""
            return content

        async def aprocess(
            self,
            message: Union[str, Dict[str, Any], Message],
            response_schema=None
        ) -> ModelResponse:
            # Get message content
            message.content if isinstance(message, Message) else str(message)

            # For testing, always return both tool calls
            return ModelResponse(
                content="Delegating to multiple members",
                tool_calls=[
                    {
                        "function": {
                            "name": "researcher",
                            "arguments": json.dumps({"query": "analyze data"})
                        }
                    },
                    {
                        "function": {
                            "name": "writer",
                            "arguments": json.dumps({"content": "format findings"})
                        }
                    }
                ],
                model="gpt-4o-mini",
                role=Role.ASSISTANT
            )

    team = Team(
        name="MultiDelegationTeam",
        leader=MultiDelegationLeader(),
        members={
            "researcher": Researcher(),
            "writer": Writer()
        }
    )

    response = await team.aprocess(Message(content="Handle this task"))
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 2
    assert all("result" in call for call in response.tool_calls)

@pytest.mark.asyncio
async def test_team_invalid_delegation():
    """Test team handling delegation to non-existent member."""
    @agent(model="gpt-4o-mini", temperature=0)
    class InvalidDelegationLeader:
        """Leader that attempts to delegate to non-existent member."""

        @tool
        def nonexistent(
            self,
            test: Annotated[str, Field(description="Test parameter")]
        ) -> str:
            """Attempt to delegate to non-existent member."""
            return test

        async def aprocess(
            self,
            message: Union[str, Dict[str, Any], Message],
            response_schema=None
        ) -> ModelResponse:
            # Get message content
            message.content if isinstance(message, Message) else str(message)

            # For testing, always try to use non-existent tool
            return ModelResponse(
                content="Delegating to invalid member",
                tool_calls=[{
                    "function": {
                        "name": "nonexistent",
                        "arguments": json.dumps({"test": "test"})
                    }
                }],
                model="gpt-4o-mini",
                role=Role.ASSISTANT
            )

    team = Team(
        name="InvalidDelegationTeam",
        leader=InvalidDelegationLeader(),
        members={
            "researcher": Researcher()
        }
    )

    response = await team.aprocess(Message(content="test"))
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert "result" not in response.tool_calls[0]

@pytest.mark.asyncio
async def test_team_context_in_delegations(basic_team):
    """Test that context is properly passed in delegations."""
    message = Message(
        content="Analyze with context",
        context={"key": "value"}
    )

    response = await basic_team.aprocess(message)
    assert response.tool_calls is not None
    assert response.tool_calls[0]["result"]

    # Verify delegation was recorded
    assert len(basic_team._delegation_history) == 1
    delegation = basic_team._delegation_history[0]
    assert delegation["source"] == "TeamLeader"
    assert delegation["target"] == "researcher"
    assert delegation["task"] == json.dumps({"query": message.content})

@pytest.mark.asyncio
async def test_team_empty_message():
    """Test team handling empty message."""
    team = Team(
        name="EmptyMessageTeam",
        leader=TeamLeader(),
        members={"researcher": Researcher()}
    )

    response = await team.aprocess(Message(content=""))
    assert isinstance(response, ModelResponse)
    assert response.content

@pytest.mark.asyncio
async def test_team_complex_interaction():
    """Test a more complex team interaction with multiple steps."""
    @agent(model="gpt-4o-mini", temperature=0)
    class ComplexLeader:
        """Leader that handles complex interactions."""

        @tool
        def researcher(
            self,
            query: Annotated[str, Field(description="The information to analyze")]
        ) -> str:
            """Delegate research tasks."""
            return query

        @tool
        def writer(
            self,
            content: Annotated[str, Field(description="The content to format")]
        ) -> str:
            """Delegate writing tasks."""
            return content

        async def aprocess(
            self,
            message: Union[str, Dict[str, Any], Message],
            response_schema=None
        ) -> ModelResponse:
            # Get message content
            msg_content = message.content if isinstance(message, Message) else str(message)

            # For testing, use deterministic tool selection based on message
            if "research" in msg_content.lower():
                return ModelResponse(
                    content="Delegating research",
                    tool_calls=[{
                        "function": {
                            "name": "researcher",
                            "arguments": json.dumps({"query": msg_content})
                        }
                    }],
                    model="gpt-4o-mini",
                    role=Role.ASSISTANT
                )
            else:
                return ModelResponse(
                    content="Delegating writing",
                    tool_calls=[{
                        "function": {
                            "name": "writer",
                            "arguments": json.dumps({"content": msg_content})
                        }
                    }],
                    model="gpt-4o-mini",
                    role=Role.ASSISTANT
                )

    team = Team(
        name="ComplexTeam",
        leader=ComplexLeader(),
        members={
            "researcher": Researcher(),
            "writer": Writer()
        }
    )

    # Test research delegation
    research_response = await team.aprocess(
        Message(content="Please research this topic")
    )
    assert research_response.tool_calls is not None
    assert research_response.tool_calls[0]["function"]["name"] == "researcher"

    # Test writing delegation
    write_response = await team.aprocess(
        Message(content="Please write this up")
    )
    assert write_response.tool_calls is not None
    assert write_response.tool_calls[0]["function"]["name"] == "writer"

if __name__ == "__main__":
    # Configure pytest arguments
    args = [
        __file__,
        "-v",
        "-p", "no:warnings",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    sys.exit(pytest.main(args))
