"""Offline tests for the optional Parallel example and Legion dispatch."""

import asyncio
import json
from functools import partial
from typing import Dict, List, Optional
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("mcp")

from examples.tools import parallel_search as example
from legion.agents.base import Agent
from legion.errors import ToolError
from legion.interface.schemas import ModelResponse, Role


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Dict:
    """Exercise the real MCP SDK against a recorded HTTP transport."""
    requests: List[httpx.Request] = []
    clients: List[httpx.AsyncClient] = []
    state = {
        "requests": requests, "clients": clients, "error": False,
        "block": False, "started": asyncio.Event(),
    }

    async def handle(request: httpx.Request) -> httpx.Response:
        """Return protocol-valid discovery and tool responses."""
        requests.append(request)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {
                "protocolVersion": "2025-03-26", "capabilities": {},
                "serverInfo": {"name": "test-parallel", "version": "1"},
            }
        else:
            assert message["method"] == "tools/call"
            if state["block"]:
                state["started"].set()
                await asyncio.Event().wait()
            result = {
                "content": [{"type": "text", "text": (
                    "rate limited" if state["error"]
                    else "https://docs.python.org/3/library/asyncio.html "
                         "asyncio provides concurrent code using async/await"
                )}],
                "isError": state["error"],
            }
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": message["id"], "result": result,
        })

    def client_factory(
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[httpx.Timeout] = None,
        auth: Optional[httpx.Auth] = None,
    ) -> httpx.AsyncClient:
        """Preserve actual SDK headers, authentication and timeouts."""
        client = httpx.AsyncClient(
            headers=headers, timeout=timeout, auth=auth,
            transport=httpx.MockTransport(handle),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(example, "streamablehttp_client", partial(
        example.streamablehttp_client, httpx_client_factory=client_factory,
    ))
    # Environment and saved credentials must not authenticate this example.
    monkeypatch.setenv("PARALLEL_API_KEY", "unused-sentinel")
    return state


@pytest.mark.asyncio
async def test_agent_dispatch_and_fetch(
    server: Dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model-selected Legion tool returns useful output into memory."""
    provider = AsyncMock()
    provider.acomplete.return_value = ModelResponse(
        content="Searching", tool_calls=[{
            "id": "search-1", "type": "function", "function": {
                "name": "parallel_search", "arguments": json.dumps({
                    "objective": "Find Python asyncio documentation",
                    "search_queries": ["Python asyncio documentation"],
                }),
            },
        }],
    )
    monkeypatch.setattr(Agent, "_setup_provider", lambda *args: provider)
    async with example.parallel_tools(session_id="test-conversation") as tools:
        researcher = Agent(name="researcher", model="openai:gpt-4o-mini")
        researcher.tools = tools
        schema = tools[0].get_schema()["function"]
        assert schema["name"] == "parallel_search"
        assert schema["parameters"]["required"] == [
            "objective", "search_queries",
        ]
        assert "session_id" not in schema["parameters"]["properties"]
        await researcher.aprocess("Find asyncio docs")
        outputs = [
            m for m in researcher.memory.messages if m.role == Role.TOOL
        ]
        assert len(outputs) == 1
        assert "https://docs.python.org" in outputs[0].content
        assert "async/await" in outputs[0].content
        assert "async/await" in await tools[1](
            urls=["https://docs.python.org/3/library/asyncio.html"],
        )
        with pytest.raises(ToolError, match="Invalid parameters"):
            await tools[0](objective="test", search_queries=[])
    calls = [json.loads(r.content) for r in server["requests"]]
    tool_calls = [m["params"] for m in calls if m["method"] == "tools/call"]
    assert [c["name"] for c in tool_calls] == ["web_search", "web_fetch"]
    assert all(c["arguments"]["session_id"] == "test-conversation"
               for c in tool_calls)
    for request in server["requests"]:
        assert str(request.url) == example.MCP_URL
        assert request.headers["User-Agent"] == example.USER_AGENT
        assert "Authorization" not in request.headers
        assert request.extensions["timeout"]["connect"] == 30
        assert request.extensions["timeout"]["read"] == 60
    assert all(client.is_closed for client in server["clients"])


@pytest.mark.asyncio
async def test_server_error_and_cleanup(server: Dict) -> None:
    """MCP error content raises a Legion error and closes the connection."""
    server["error"] = True
    async with example.parallel_tools() as (search, _):
        with pytest.raises(ToolError, match="rate limited"):
            await search(objective="test", search_queries=["test query"])
    assert all(client.is_closed for client in server["clients"])


@pytest.mark.asyncio
async def test_cancellation_closes_connection(server: Dict) -> None:
    """Cancellation of a conversation closes the SDK's HTTP client."""
    server["block"] = True

    async def conversation() -> None:
        """Keep the connection open until the caller cancels."""
        async with example.parallel_tools() as (search, _):
            await search(objective="test", search_queries=["test query"])

    task = asyncio.create_task(conversation())
    await asyncio.wait_for(server["started"].wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(client.is_closed for client in server["clients"])
