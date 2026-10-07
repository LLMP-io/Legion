# Parallel Search MCP tools

This example wraps Parallel's anonymous Search MCP endpoint in Legion's
`@tool` interface. It provides `parallel_search` and `parallel_fetch` over
Streamable HTTP at `https://search.parallel.ai/mcp`. No Parallel API key is
needed. The free endpoint has rate limits and server-managed search settings;
see the [Search MCP documentation](https://docs.parallel.ai/integrations/mcp/search-mcp).

From the repository root, install in a fresh Python 3.11 environment:

```bash
python3.11 -m venv .venv-parallel
. .venv-parallel/bin/activate
pip install -r examples/tools/requirements-parallel.txt
python -m examples.tools.parallel_search "Find Python asyncio documentation" \
  --query "Python asyncio documentation" \
  --fetch https://docs.python.org/3/library/asyncio.html
```

The command calls the Legion tools directly and prints source URLs and excerpts.
It needs no model credentials and does not run model inference. Fetch output
also preserves any per-URL errors returned by the server. MCP errors propagate
as Legion `ToolError`; connection timeouts are 30 seconds and read/tool-response
timeouts are 60 seconds. Cancellation and connection cleanup use the MCP SDK's async
context managers. The SDK is pinned for compatibility with Legion's Pydantic
version. Existing provider defaults and dependencies remain unchanged.

To use the tools with an existing Legion agent:

```python
from examples.tools.parallel_search import parallel_tools

async with parallel_tools() as tools:
    researcher.tools = tools  # An existing Legion Agent instance.
    await researcher.aprocess("Find current Python asyncio documentation")
    # Legion records tool output in researcher.memory.messages.
```

Keep the context open for the entire conversation; these tools use its active
MCP session. Each context generates one stable conversation identifier shared
by search and fetch. Pass `session_id` to reuse that identifier when reconnecting
for the same conversation. Starting another conversation should use a new ID.
The example sends a Legion User-Agent and no Authorization header, regardless
of saved or environment credentials. An agent still needs its own model setup;
model inference costs and limits are separate from anonymous search access.
Legion's agent returns the model response and stores tool results in memory;
it does not automatically produce another model response after the tool call.

The example's offline tests use the real MCP SDK with a local HTTP fixture and
a deterministic model response to check Legion agent dispatch, request headers,
errors, and cleanup:

```bash
pip install pytest pytest-asyncio
python -m pytest examples/tools/test_parallel_search.py
```
