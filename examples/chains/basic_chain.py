"""Basic Chain Example

This example demonstrates how to create a simple chain of two agents using Legion's
decorator syntax. The chain processes text through:
1. A summarizer that condenses the input
2. An analyzer that provides insights about the summary
"""

from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field

from legion import agent, chain, tool

load_dotenv()


# First agent: Summarizes text
@agent(
    model="openai:gpt-4o-mini",
    temperature=0.3  # Lower temperature for more consistent summaries
)
class Summarizer:
    """I am a text summarizer that creates concise summaries while preserving key information.
    I aim to reduce text length by 70-80% while maintaining the most important points.
    """

    @tool
    def count_words(self, text: Annotated[str, Field(description="Text to count words in")]) -> int:
        """Count the number of words in a text"""
        return len(text.split())


# Second agent: Analyzes summaries
@agent(
    model="openai:gpt-4o-mini",
    temperature=0.7  # Higher temperature for more creative analysis
)
class Analyzer:
    """I am a text analyzer that provides insights about the content.
    I focus on identifying key themes, tone, and potential implications.
    """

    @tool
    def identify_keywords(
        self,
        text: Annotated[str, Field(description="Text to extract keywords from")]
    ) -> list[str]:
        """Extract main keywords from text"""
        # This is just a simple example - in practice you might use NLP
        words = text.lower().split()
        return list(set(w for w in words if len(w) > 5))[:5]  # Just a simple example


# Create a chain that combines both agents
@chain
class TextAnalysisChain:
    """A chain that first summarizes text and then analyzes the summary."""

    # Define the agents in the order they should process
    summarizer = Summarizer()
    analyzer = Analyzer()
    
    # Define the members list for the chain
    members = [Summarizer(), Analyzer()]


async def main():
    # Create an instance of our chain
    processor = TextAnalysisChain(verbose=True)  # Enable verbose output to see chain progress

    # Example text to process
    long_text = """
    Artificial Intelligence has transformed numerous industries in recent years.
    From healthcare to finance, AI systems are being deployed to automate tasks,
    analyze complex data, and make predictions. In healthcare, AI helps doctors
    diagnose diseases and plan treatments. In finance, it detects fraudulent
    transactions and predicts market trends. However, these advancements also
    raise important ethical considerations about privacy, bias, and the role of
    human oversight in AI-driven decisions. As we continue to develop more
    sophisticated AI systems, addressing these ethical concerns becomes increasingly
    crucial for responsible innovation.
    """

    print("Original Text:")
    print(long_text.strip())
    print("\n" + "=" * 50 + "\n")

    # Process the text through our chain
    response = await processor.aprocess(long_text)

    print("\nFinal Chain Output:")
    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
