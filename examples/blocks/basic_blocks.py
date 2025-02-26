"""Basic Blocks Example

This example demonstrates how to create and use blocks in Legion.
Blocks are enhanced functions that provide:
- Input/output validation
- Execution monitoring
- Async compatibility
- Chain integration
"""

import logging
from typing import Any, Dict, List  # noqa: F401

from pydantic import BaseModel, Field

from legion import block, chain

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define input/output schemas for type safety
class TextInput(BaseModel):
    text: str = Field(description="Input text to process")


class WordCountOutput(BaseModel):
    word_count: int = Field(description="Number of words in text")
    char_count: int = Field(description="Number of characters in text")


class SentimentOutput(BaseModel):
    sentiment: str = Field(description="Detected sentiment (positive/negative/neutral)")
    confidence: float = Field(description="Confidence score of sentiment")


# Create a simple word counter block using decorator syntax
@block(
    input_schema=TextInput,
    output_schema=WordCountOutput,
    tags=["text", "analysis"]
)
def count_words(input_data: TextInput) -> WordCountOutput:
    """Count words and characters in text."""
    text = input_data.text
    words = len(text.split())
    chars = len(text)
    return WordCountOutput(word_count=words, char_count=chars)


# Create a mock sentiment analysis block using decorator syntax
@block(
    input_schema=TextInput,
    output_schema=SentimentOutput,
    tags=["text", "sentiment", "nlp"]
)
async def analyze_sentiment(input_data: TextInput) -> SentimentOutput:
    """Analyze sentiment of text (mock implementation)."""
    # In real usage, this would call an NLP model
    text = input_data.text.lower()
    positive_words = {"good", "great", "excellent", "happy", "wonderful"}
    negative_words = {"bad", "terrible", "awful", "sad", "horrible"}

    pos_count = sum(1 for word in text.split() if word in positive_words)
    neg_count = sum(1 for word in text.split() if word in negative_words)

    if pos_count > neg_count:
        return SentimentOutput(sentiment="positive", confidence=0.8)
    elif neg_count > pos_count:
        return SentimentOutput(sentiment="negative", confidence=0.8)
    return SentimentOutput(sentiment="neutral", confidence=0.6)


# Define the chain using decorator syntax
@chain
class TextAnalysisChain:
    """A chain that analyzes text by counting words and determining sentiment."""

    # List the blocks in processing order
    members = [
        count_words,
        analyze_sentiment
    ]


async def main():
    # Create chain instance
    text_analysis_chain = TextAnalysisChain()

    # Process some text
    text = "This is a great example of using Legion blocks!"
    input_data = TextInput(text=text)

    print(f"\nAnalyzing text: '{text}'\n")

    # Execute individual blocks
    print("Individual block execution:")
    word_count_result = await count_words(input_data)
    print(f"Word count result: {word_count_result}")

    sentiment_result = await analyze_sentiment(input_data)
    print(f"Sentiment result: {sentiment_result}\n")

    # Execute chain
    print("Chain execution:")
    chain_result = await text_analysis_chain.aprocess(input_data)
    print(f"Chain results: {chain_result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
