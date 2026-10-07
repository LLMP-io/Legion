"""Use anonymous Parallel Search MCP through Legion's tool interface."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncIterator, List, Optional
from uuid import uuid4

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from legion import tool
from legion.errors import ToolError
from legion.interface.tools import BaseTool

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "Legion/0.1.5 (Parallel Search MCP example)"


def result_text(result: CallToolResult) -> str:
    """Keep server text, including source URLs and partial fetch errors."""
    text = "\n".join(
        item.text for item in result.content if isinstance(item, TextContent)
    )
    if result.isError:
        raise ToolError(text or "Parallel MCP tool call failed")
    if not text:
        raise ToolError("Parallel MCP returned no text content")
    return text


@asynccontextmanager
async def parallel_tools(
    session_id: Optional[str] = None,
) -> AsyncIterator[List[BaseTool]]:
    """Keep one anonymous MCP connection alive for a conversation."""
    conversation_id = session_id or str(uuid4())
    async with streamablehttp_client(
        MCP_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=timedelta(seconds=30),
        sse_read_timeout=timedelta(seconds=60),
    ) as (read, write, _):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()

            @tool
            async def parallel_search(
                objective: str,
                search_queries: List[str] = Field(
                    ..., min_length=1,
                    description="Related keyword queries, 3-6 words each",
                ),
            ) -> str:
                """Search the web and return source URLs and excerpts."""
                result = await session.call_tool("web_search", {
                    "objective": objective,
                    "search_queries": search_queries,
                    "session_id": conversation_id,
                })
                return result_text(result)

            @tool
            async def parallel_fetch(
                urls: List[str] = Field(
                    ..., min_length=1, description="Web URLs to read",
                ),
                objective: Optional[str] = None,
            ) -> str:
                """Fetch relevant excerpts from specific web pages."""
                result = await session.call_tool("web_fetch", {
                    "urls": urls,
                    "objective": objective,
                    "session_id": conversation_id,
                })
                return result_text(result)

            yield [parallel_search, parallel_fetch]


async def main() -> None:
    """Run Legion tools directly without a model or API key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objective", help="What you want to find")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--fetch", action="append", help="URL to read")
    args = parser.parse_args()
    async with parallel_tools() as (search, fetch):
        print(await search(
            objective=args.objective, search_queries=args.query,
        ))
        if args.fetch:
            print(await fetch(urls=args.fetch, objective=args.objective))


if __name__ == "__main__":
    asyncio.run(main())
