"""Example demonstrating a chain with both blocks and agents.

This example shows how to combine functional blocks with LLM-powered agents
in a processing chain. The chain processes text through the following steps:
1. A block that normalizes text (removes extra whitespace, etc.)
2. An agent that summarizes the text
3. A block that extracts key metrics from the summary

Note: This example requires an OpenAI API key. Set it in your environment:
    export OPENAI_API_KEY=your_api_key_here
"""

import asyncio
import json
import os
from typing import Dict, List
from pydantic import BaseModel

from dotenv import load_dotenv

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

# Output schemas for validation
class NormalizedText(BaseModel):
    """Schema for normalized text output"""
    text: str
    char_count: int
    word_count: int

class TextMetrics(BaseModel):
    """Schema for text metrics output"""
    sentence_count: int
    avg_sentence_length: float
    key_phrases: List[str]

@block(output_schema=NormalizedText)
def normalize_text(text: str) -> Dict:
    """Block that normalizes text by cleaning whitespace and counting stats"""
    print("\nNormalize Block Input:", text[:100], "...")
    
    # Remove extra whitespace and normalize line endings
    cleaned = ' '.join(text.split())
    
    result = {
        "text": cleaned,
        "char_count": len(cleaned),
        "word_count": len(cleaned.split())
    }
    print("Normalize Block Output:", result)
    return result

@agent(
    model="openai:gpt-4o-mini",
    temperature=0.0,
    max_tokens=1000,
)
class Summarizer:
    """You are a text summarization expert. Your role is to create clear, 
            concise summaries that capture the key points while maintaining readability.
            Focus on extracting the most important information and presenting it in a 
            well-structured format.
            
            Your output should be a concise summary that:
            1. Captures the main ideas
            2. Preserves key terminology
            3. Uses clear, direct language
            4. Is roughly 1/3 the length of the input"""
    pass

@block(output_schema=TextMetrics)
def extract_metrics(text: str) -> Dict:
    """Block that extracts key metrics from text"""
    print("\nMetrics Block Input:", text[:100], "...")
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    words = text.split()
    
    # Extract key phrases (simple implementation)
    key_phrases = []
    for i in range(len(words) - 2):
        phrase = ' '.join(words[i:i+3])
        if any(w[0].isupper() for w in words[i:i+3]):
            key_phrases.append(phrase)
    
    result = {
        "sentence_count": len(sentences),
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
        "key_phrases": key_phrases[:5]  # Top 5 phrases
    }
    print("Metrics Block Output:", result)
    return result

@chain
class TextProcessor:
    """Chain that processes text through normalization, summarization, and metrics extraction"""
    
    # Define chain members in processing order
    members = [
        normalize_text,  # First normalize the text
        Summarizer,      # Then create a summary
        extract_metrics  # Finally extract metrics from the summary
    ]

async def main():
    # Create chain instance
    processor = TextProcessor(verbose=True)
    
    # Sample text to process
    text = """
    Artificial Intelligence (AI) has transformed many industries in recent years.
    Machine learning models can now perform tasks that were once thought impossible.
    Natural Language Processing has enabled computers to understand and generate human-like text.
    Deep Learning architectures have revolutionized computer vision and speech recognition.
    The future of AI looks promising, with new breakthroughs happening regularly.
    """
    
    # Process the text through the chain
    result = await processor.aprocess(text)
    metrics = json.loads(result.content)
    
    print("\nFinal Output:")
    print("Key Phrases:", metrics["key_phrases"])
    print(f"Sentence Count: {metrics['sentence_count']}")
    print(f"Average Sentence Length: {metrics['avg_sentence_length']:.1f} words")

if __name__ == "__main__":
    asyncio.run(main()) 