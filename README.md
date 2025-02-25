# Legion: A Provider-Agnostic Multi-Agent Framework

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.0.1-blue.svg)](https://github.com/og-hayden/legion)

Legion is a flexible and provider-agnostic framework designed to simplify the creation of sophisticated multi-agent systems. It provides a set of tools and abstractions to build, manage, and monitor AI agents, chains, teams, and graphs, allowing you to focus on the logic of your application rather than the underlying infrastructure.

## Key Features

*   **Provider Agnostic:** Supports multiple LLM providers (OpenAI, Anthropic, Groq, Ollama, Gemini) through a unified interface, allowing you to switch or combine providers easily.
*   **Agent Abstraction:** Define agents with clear roles, tools, and system prompts using decorators, simplifying the creation of complex agent behaviors.
*   **Tool Integration:** Seamlessly integrate tools with agents using decorators, enabling agents to interact with external systems and data.
*   **Parameter Injection:** Inject parameters into tools at runtime, allowing for dynamic configuration and secure handling of sensitive credentials.
*   **Chains and Teams:** Build complex workflows by chaining agents and blocks together or creating collaborative teams of agents.
*   **Graph-Based Execution:** Construct complex processing graphs with nodes, edges, and channels, enabling flexible and scalable workflows.
*   **Input/Output Validation:** Ensure data integrity with Pydantic-based input and output schemas for both agents and blocks.
*   **Dynamic System Prompts:** Create agents with system prompts that adapt to context and user preferences.
*   **Memory Management:** Supports various memory providers for storing and retrieving conversation history and agent state.
*   **Monitoring and Observability:** Built-in monitoring tools for tracking performance, errors, and resource usage.
*   **Asynchronous Support:** Fully asynchronous design for efficient and scalable operations.

## Installation (WIP, not yet available during pre-release phase)

```bash
pip install legion-ai
```

## Core Concepts

### API Structure: Decorators and Base Classes

Legion provides a dual-layer API structure:

1. **Simplified Decorator API** (Top-level imports):
   ```python
   from legion import agent, block, chain, tool
   ```
   These decorators provide a quick, intuitive way to create components with minimal boilerplate. They're perfect for most use cases and are imported directly from the top-level `legion` package.

2. **Advanced Base Class API** (Module-level imports):
   ```python
   from legion.agents.base import Agent
   from legion.groups.chain import Chain
   from legion.interface.tools import BaseTool
   ```
   These base classes provide more control and customization options. They're useful when you need to extend functionality or have specific requirements that the decorators don't cover.

This dual-layer approach gives you both simplicity for common use cases and flexibility for advanced scenarios.

### Agents

Agents are the fundamental building blocks of Legion. They are defined using the `@agent` decorator and can have:

*   A model (e.g., `openai:gpt-4o-mini`)
*   A temperature
*   A system prompt (static or dynamic)
*   A set of tools

```python
from typing import List, Annotated
from pydantic import Field
from legion.agents import agent
from legion.interface.decorators import tool

@tool
def add_numbers(numbers: Annotated[List[float], Field(description="List of numbers to add")]) -> float:
    """Add a list of numbers together"""
    return sum(numbers)

@agent(model="openai:gpt-4o-mini", temperature=0.2, tools=[add_numbers])
class MathHelper:
    """You are a helpful assistant that helps with basic math operations"""

    @tool
    def format_result(self, number: Annotated[float, Field(description="Number to format")], prefix: str = "Result: ") -> str:
        """Format a number with a custom prefix"""
        return f"{prefix}{number:.2f}"

async def main():
    agent = MathHelper()
    response = await agent.aprocess("Add 1.5, 2.5, and 3.5 and format the result.")
    print(response.content)
```

### Tools

Tools are functions that agents can use to interact with the world. They are defined using the `@tool` decorator and can have:

*   A name
*   A description
*   Parameters with type hints and descriptions
*   Injectable parameters for dynamic configuration

```python
from typing import Annotated
from pydantic import Field
from legion.interface.decorators import tool

@tool(
    inject=["api_key", "endpoint"],
    description="Process a query using an external API",
    defaults={"api_key": "sk_test_default", "endpoint": "https://api.example.com/v1"}
)
def process_api_query(
    query: Annotated[str, Field(description="The query to process")],
    api_key: Annotated[str, Field(description="API key for the service")],
    endpoint: Annotated[str, Field(description="API endpoint")]
) -> str:
    """Process a query using an external API"""
    # API call logic here
    return f"Processed '{query}' using API at {endpoint} with key {api_key[:4]}..."
```

### Chains

Chains are sequences of agents or blocks that process data sequentially. They are defined using the `@chain` decorator:

```python
from legion.agents import agent
from legion.groups.decorators import chain
from legion.interface.decorators import tool

@agent(model="openai:gpt-4o-mini", temperature=0.3)
class Summarizer:
    """I am a text summarizer"""
    @tool
    def count_words(self, text: str) -> int:
        return len(text.split())

@agent(model="openai:gpt-4o-mini", temperature=0.7)
class Analyzer:
    """I am a text analyzer"""
    @tool
    def identify_keywords(self, text: str) -> list[str]:
        words = text.lower().split()
        return list(set(w for w in words if len(w) > 5))[:5]

@chain
class TextAnalysisChain:
    """A chain that summarizes text and then analyzes the summary"""
    summarizer = Summarizer()
    analyzer = Analyzer()

async def main():
    processor = TextAnalysisChain(verbose=True)
    response = await processor.aprocess("Long text here...")
    print(response.content)
```

### Teams

Teams are groups of agents that collaborate on tasks. They are defined using the `@team` decorator and include a leader agent and member agents:

```python
from typing import List, Dict, Any, Annotated
from legion.agents.decorators import agent
from legion.groups.decorators import team, leader
from legion.interface.decorators import tool
from pydantic import Field

@tool
def analyze_numbers(numbers: Annotated[List[float], Field(description="List of numbers to analyze")]) -> Dict[str, float]:
    """Analyze a list of numbers and return basic statistics"""
    if not numbers:
        return {"mean": 0, "min": 0, "max": 0}
    return {
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }

@tool
def format_report(title: Annotated[str, Field(description="Report title")], sections: Annotated[List[str], Field(description="List of report sections")]) -> str:
    """Format a professional report with sections"""
    parts = [f"# {title}\\n"]
    for i, section in enumerate(sections, 1):
        parts.extend([f"\\n## Section {i}", section])
    return "\\n".join(parts)

@team
class ResearchTeam:
    """A team that collaborates on research tasks."""

    @leader(model="openai:gpt-4o-mini", temperature=0.2)
    class Leader:
        """Research team coordinator who delegates tasks and synthesizes results."""
        pass

    @agent(model="openai:gpt-4o-mini", temperature=0.1, tools=[analyze_numbers])
    class Analyst:
        """Data analyst who processes numerical data and provides statistical insights."""

    @agent(model="openai:gpt-4o-mini", temperature=0.7, tools=[format_report])
    class Writer:
        """Technical writer who creates clear and professional reports."""

async def main():
    team = ResearchTeam()
    response = await team.aprocess("Analyze these numbers: [10.5, 20.5, 15.0, 30.0, 25.5] and create a report.")
    print(response.content)
```

### Graphs

Graphs allow you to create complex workflows with nodes, edges, and channels. Nodes can be agents, blocks, or other graphs. Edges connect nodes and define the flow of data. Channels are used for data transfer between nodes.

```python
import asyncio
from typing import List, Annotated
from pydantic import BaseModel, Field

from legion.graph.decorators import graph
from legion.graph.nodes.decorators import node
from legion.interface.decorators import tool
from legion.agents.decorators import agent
from legion.blocks.decorators import block
from legion.graph.edges.base import EdgeBase
from legion.graph.channels import LastValue

class TextData(BaseModel):
    text: str = Field(description="Input text to process")

@block(input_schema=TextData, output_schema=TextData, tags=["text", "preprocessing"])
def normalize_text(input_data: TextData) -> TextData:
    """Normalize text by removing extra whitespace."""
    text = ' '.join(input_data.text.split())
    return TextData(text=text)

@agent(model="openai:gpt-4o-mini", temperature=0.2)
class Summarizer:
    """An agent that summarizes text."""
    @tool
    def count_words(self, text: Annotated[str, Field(description="Text to count words in")]) -> int:
        """Count the number of words in a text"""
        return len(text.split())

class TextEdge(EdgeBase):
    """Edge for connecting text processing nodes"""
    pass

@graph(name="basic_text_processing", description="A simple graph for processing text")
class TextProcessingGraph:
    """A graph that first normalizes text and then summarizes it."""
    normalizer = normalize_text
    summarizer = Summarizer()

    edges = [
        {
            "edge_type": TextEdge,
            "source_node": "normalizer",
            "target_node": "summarizer",
            "source_channel": "output",
            "target_channel": "input"
        }
    ]

    input_channel = LastValue(type_hint=str)
    output_channel = LastValue(type_hint=str)

    async def process(self, input_text: str) -> str:
        """Process text through the graph"""
        self.input_channel.set(input_text)
        await self.graph.execute()
        return self.output_channel.get()

async def main():
    text_graph = TextProcessingGraph()
    input_text = "  This    is   a   test   with    extra   spaces.  "
    output_text = await text_graph.process(input_text)
    print(f"Original Text: '{input_text}'")
    print(f"Processed Text: '{output_text}'")
```

### Blocks

Blocks are functional units that can be used in chains or graphs. They are defined using the `@block` decorator and provide input/output validation.

```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from legion.blocks import block

class TextInput(BaseModel):
    text: str = Field(description="Input text to process")

class WordCountOutput(BaseModel):
    word_count: int = Field(description="Number of words in text")
    char_count: int = Field(description="Number of characters in text")

@block(input_schema=TextInput, output_schema=WordCountOutput, tags=["text", "analysis"])
def count_words(input_data: TextInput) -> WordCountOutput:
    """Count words and characters in text."""
    text = input_data.text
    words = len(text.split())
    chars = len(text)
    return WordCountOutput(word_count=words, char_count=chars)
```

## Examples

The `examples` directory contains several examples that demonstrate how to use Legion's features:

*   `examples/agents/basic_agent.py`: Shows how to create a basic agent with internal and external tools.
*   `examples/agents/schema_agent.py`: Demonstrates how to use output schemas with agents.
*   `examples/agents/dynamic_prompt_agent.py`: Shows how to create an agent with a dynamic system prompt.
*   `examples/blocks/basic_blocks.py`: Illustrates how to create and use functional blocks.
*   `examples/chains/basic_chain.py`: Shows how to create a simple chain of agents.
*   `examples/chains/mixed_chain.py`: Demonstrates a chain with both blocks and agents.
*   `examples/graph/basic_graph.py`: Shows how to create a simple graph with nodes and edges.
*   `examples/teams/basic_team.py`: Demonstrates how to create a team of agents.
*   `examples/tools/basic_tools.py`: Shows how to create and use tools with an agent.
*   `examples/tools/injected_tools.py`: Demonstrates parameter injection with tools.

## Setting Up the Development Environment

### Option 1: Using `Makefile`
Optioned and automated
- Requires you have `make` and `poetry` installed
- `ENV` management options are currently `venv` or `conda`
- `POETRY` can be set to `true` or `false` to use it, or `pip`

2a. [Install Make](https://www.gnu.org/software/make/manual/make.html)

2b. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2c. Set up the environment

```bash
make setup ENV=venv POETRY=false
# or
make setup ENV=conda POETRY=true
# or
make  # Just use the defaults
```
This will:
- Create and activate a virtual environment (with specified `ENV`)
- Install all dependencies (with `pip` or `poetry`)
- Set up pre-commit hooks

You can also:
```bash
make test POETRY=<true|false>
```

### Option 2: Using `setup.py`
Default and standard `venv`/`pip` setup
```bash
# Run the setup script
python3 scripts/setup_env.py

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```
This will:
- Create and activate a virtual environment
- Install all dependencies
- Set up pre-commit hooks

### Option 3: 🐳 Whale you can just use `Docker`
If you have Docker installed and the docker engine running, from the
project root you can just run:
```bash
docker compose up --build
```
This will spin up a cointainer, build the project to it, and run the tests.
Want to keep it running and get a shell on it to run other commands?
```bash
docker compose up -d  # detached
docker compose exec legion bash
```


## Documentation

For more detailed information, please refer to the docstrings within the code itself.
Eventually, there will be a more comprehensive documentation site.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report issues, or suggest new features.

## Authors

- Hayden Smith (hayden@llmp.io)
- Zain Imdad (zain@llmp.io)

## Project Status & Roadmap

View the [Legion Project Board](https://github.com/orgs/LLMP-io/projects/1/views/1) for the current status of the project.

To view the current roadmap for Legion, please see the [Legion Roadmap](https://github.com/orgs/LLMP-io/projects/1/views/4).

## License

Legion is released under the MIT License. See the `LICENSE` file for more details.
