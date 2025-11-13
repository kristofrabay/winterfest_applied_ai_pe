"""
User Memory Layer - Extract primary and secondary intents from user text using OpenAI structured outputs.
"""

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from typing import List
import json


class UserMemory(BaseModel):
    """Structured model for user intents extracted from text."""
    
    primary_intent: str = Field(
        description="The most important, high-level intent or need of the user. What is their main goal or concern?"
    )
    
    secondary_intents: List[str] = Field(
        description="Additional, episodic intents or contextual details that support understanding the user",
        default_factory=list
    )


async def extract_user_memory(
    user_text: str,
    model: str = "gpt-4o-mini",
    client: AsyncOpenAI = None,
) -> UserMemory:
    """
    Extract user memory (primary and secondary intents) from text using OpenAI structured outputs.
    
    Args:
        user_text: The text input from the user to analyze
        api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
        model: OpenAI model to use (must support structured outputs)
    
    Returns:
        UserMemory object with extracted intents
    """
    
    system_prompt = """You are an expert at understanding user intent and building user profiles.
Analyze the provided text and extract:
1. PRIMARY INTENT: The most important, high-level thing about the user - their main goal, need, or defining characteristic
2. SECONDARY INTENTS: Supporting details, episodic information, or contextual elements that help describe the user

Be concise and insightful. Focus on what matters most to understand and serve the user."""

    user_prompt = f"Analyze this text and extract user intents:\n\n{user_text}"
    
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=UserMemory,
    )
    
    return completion.choices[0].message.parsed


