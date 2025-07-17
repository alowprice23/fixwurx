#!/usr/bin/env python3
"""
Real LLM Client

This module provides a client for interacting with OpenAI's GPT-4o model.
It implements a full integration with the OpenAI API while keeping the mock
implementation available for testing purposes.
"""

import os
import sys
import json
import time
import logging
import dotenv
import openai
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_client.log")
    ]
)

logger = logging.getLogger("LLMClientReal")

class LLMClientReal:
    """
    Client for interacting with OpenAI's GPT models.
    
    This class provides methods for:
    - Generating text completions
    - Managing conversation context
    - Handling API communication
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.api_key = self.config.get("api_key", os.getenv("OPENAI_API_KEY", ""))
        self.model = self.config.get("model", os.getenv("DEFAULT_MODEL", "gpt-4o"))
        self.max_tokens = int(self.config.get("max_tokens", os.getenv("MAX_TOKENS", 4096)))
        self.temperature = float(self.config.get("temperature", os.getenv("TEMPERATURE", 0.2)))
        self.initialized = False
        self.use_mock = not self.api_key  # Use mock when API key is not available
        # Add response cache to avoid redundant API calls
        self.response_cache = {}
        self.max_cache_size = int(self.config.get("max_cache_size", 100))
        
        # Initialize OpenAI client if API key is available
        if self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.use_mock = True  # Fall back to mock if initialization fails
        else:
            logger.warning("No OpenAI API key provided, falling back to mock implementation")
            self.use_mock = True
        
        # Initialize mock client for fallback
        if self.use_mock:
            from llm_client import LLMClient
            self.mock_client = LLMClient(self.config)
            logger.info("Mock LLM client initialized as fallback")
        
        self.initialized = True
        logger.info(f"LLM Client initialized with {'real' if not self.use_mock else 'mock'} implementation")
    
    def _get_cache_key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate a unique cache key for a prompt with specific parameters"""
        key_data = f"{prompt}|{temperature}|{max_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if response is in cache and return it if found"""
        if cache_key in self.response_cache:
            logger.info(f"Cache hit for key {cache_key[:8]}...")
            return self.response_cache[cache_key]
        return None
        
    def _update_cache(self, cache_key: str, response: str) -> None:
        """Update cache with new response"""
        # Implement LRU cache by removing oldest entry if at capacity
        if len(self.response_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = response
    
    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Prompt string
            temperature: Temperature parameter for generation randomness
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text string
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        # Check cache first to avoid redundant API calls
        cache_key = self._get_cache_key(prompt, temperature, max_tokens)
        cached_response = self._check_cache(cache_key)
        if cached_response:
            return cached_response
            
        logger.info(f"Generating text with temperature={temperature}, max_tokens={max_tokens}")
        
        if self.use_mock:
            logger.info("Using mock implementation for generation")
            response = self.mock_client.generate(prompt, temperature, max_tokens)
            self._update_cache(cache_key, response)
            return response
        
        try:
            # Use messages format with the OpenAI chat completion API
            messages = [
                {"role": "system", "content": "You are a helpful assistant for the FixWurx system."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            self._update_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            
            # Fall back to mock implementation in case of errors
            logger.info("Falling back to mock implementation after API error")
            if hasattr(self, 'mock_client'):
                return self.mock_client.generate(prompt, temperature, max_tokens)
            else:
                return f"Error generating text: {e}"

    def _get_chat_cache_key(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate a unique cache key for chat messages with specific parameters"""
        messages_str = json.dumps(messages, sort_keys=True)
        key_data = f"{messages_str}|{temperature}|{max_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a chat completion based on a series of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Temperature parameter for generation randomness
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with response information
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        # Check cache first to avoid redundant API calls
        cache_key = self._get_chat_cache_key(messages, temperature, max_tokens)
        cached_response = self._check_cache(cache_key)
        if cached_response:
            return json.loads(cached_response)
            
        logger.info(f"Generating chat completion with temperature={temperature}, max_tokens={max_tokens}")
        
        if self.use_mock:
            logger.info("Using mock implementation for chat completion")
            response = self.mock_client.chat(messages, temperature, max_tokens)
            self._update_cache(cache_key, json.dumps(response))
            return response
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Convert to dictionary format similar to mock responses
            result = {
                "id": response.id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Cache the result
            self._update_cache(cache_key, json.dumps(result))
            return result
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            
            # Fall back to mock implementation in case of errors
            logger.info("Falling back to mock implementation after API error")
            if hasattr(self, 'mock_client'):
                response = self.mock_client.chat(messages, temperature, max_tokens)
                self._update_cache(cache_key, json.dumps(response))
                return response
            else:
                return {
                    "id": f"error-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Error generating chat completion: {e}"
                            },
                            "finish_reason": "error"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }

# Helper function to get the singleton instance
_instance = None

def get_instance(config: Optional[Dict[str, Any]] = None) -> LLMClientReal:
    """
    Get the singleton instance of the LLM client.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LLMClientReal instance
    """
    global _instance
    if _instance is None:
        _instance = LLMClientReal(config)
    return _instance

# Example usage
if __name__ == "__main__":
    client = get_instance()
    
    # Example usage
    prompt = "Generate a short Python function to calculate the factorial of a number."
    response = client.generate(prompt)
    print("\nGeneration Response:")
    print(response)
    
    # Example chat usage
    messages = [
        {"role": "system", "content": "You are a helpful assistant for the FixWurx system."},
        {"role": "user", "content": "What's the best way to debug a Python script?"}
    ]
    chat_response = client.chat(messages)
    print("\nChat Response:")
    print(chat_response["choices"][0]["message"]["content"])
