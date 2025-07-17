"""
api/llm_integrations.py
───────────────────────
Enhanced abstraction around popular LLM back-ends with fallback capabilities:

* **OpenAI**      – ChatGPT / GPT-4 / GPT-4o  
* **Anthropic**   – Claude-3 family  
* **Local LLMs**  – Support for sensitive environments

Key features:
• Secure credential management with no hardcoded API keys
• Fallback logic with configurable retry attempts
• Accurate token counting and cost tracking
• Support for multiple model tiers (primary, fallback, explanation, offline)
• Integration with system configuration

Public surface
──────────────
```python
# Initialize with credential manager
cm = CredentialManager()
lm = LLMManager(credential_manager=cm)

# Basic usage with preferred provider
resp = lm.chat(
    role="assistant",
    content="Explain entropy in 3 sentences.",
    task_type="explain",
    complexity="low",
)

# Force specific provider
resp = lm.chat(
    role="assistant",
    content="Debug this code...",
    provider="anthropic",
    model="claude-3-opus"
)

# Print response details
print(resp.text, resp.tokens, resp.cost_usd, resp.provider, resp.model)
```
"""
from __future__ import annotations

import importlib
import os
import time
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple, Any, Union

# Import credential manager if available
try:
    from credential_manager import CredentialManager
except ImportError:
    CredentialManager = None


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Dataclasses & Protocols
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    """Standardised response container."""
    text: str
    tokens: int
    latency_ms: float
    cost_usd: float
    provider: str
    model: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RetryStrategy(Enum):
    """Strategies for retrying failed LLM calls."""
    NONE = "none"               # No retries  # type: Final
    SAME_PROVIDER = "same"      # Retry with same provider
    ALTERNATE_PROVIDER = "alt"  # Try another provider
    ALL_PROVIDERS = "all"       # Try all available providers in sequence


class BaseProvider(Protocol):
    """Interface for all LLM provider wrappers."""
    name: str

    def chat(self, role: str, content: str, **kwargs) -> LLMResponse:  # noqa: D401
        ...
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Provider Implementations
# ──────────────────────────────────────────────────────────────────────────────
class OpenAIProvider:
    """OpenAI API wrapper with enhanced features."""
    name = "openai"
    
    # Model pricing per 1K tokens (input, output)
    PRICING = {
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4o": (0.01, 0.03),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-3.5-turbo": (0.0005, 0.0015)
    }

    def __init__(
        self,
        credential_manager: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature
        self.openai = self._import_sdk()
        
        # Get API key from credential manager or environment
        if credential_manager is not None:
            self.api_key = credential_manager.get_api_key("openai")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            
        if not self.api_key:
            raise RuntimeError("OpenAI API key not available")
            
        self.openai.api_key = self.api_key
        self.tokenizer = self._get_tokenizer()

    @staticmethod
    def _import_sdk():
        try:
            mod = importlib.import_module("openai")  # raises if not installed
            return mod
        except ImportError:
            raise RuntimeError("OpenAI SDK not installed. Install with: pip install openai")

    def _get_tokenizer(self):
        """Get tokenizer for token counting."""
        try:
            import tiktoken
            return tiktoken.encoding_for_model(self.model)
        except ImportError:
            # Fallback to word-based estimation if tiktoken not available
            return None

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using the appropriate tokenizer."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: rough estimate based on words (not accurate but better than nothing)
        return len(text.split()) + len(text) // 4

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage and model pricing."""
        model_pricing = self.PRICING.get(self.model, (0.01, 0.03))  # Default to GPT-4 pricing
        input_cost, output_cost = model_pricing
        
        return (prompt_tokens * input_cost + completion_tokens * output_cost) / 1000

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ chat
    def chat(self, role: str, content: str, **kwargs) -> LLMResponse:
        # Override model if specified in kwargs
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        
        start = time.perf_counter()
        
        try:
            # Configure client
            client = self.openai.OpenAI(api_key=self.api_key)
            
            # Count tokens for input
            prompt_tokens = self.count_tokens(content)
            
            # Create chat completion
            rsp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": role, "content": content}],
            )
            
            # Calculate metrics
            dur = (time.perf_counter() - start) * 1000
            choice = rsp.choices[0].message.content
            usage = rsp.usage
            completion_tokens = usage.completion_tokens
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            
            return LLMResponse(
                text=choice,
                tokens=usage.total_tokens,
                latency_ms=dur,
                cost_usd=cost,
                provider=self.name,
                model=model,
                metadata={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "finish_reason": rsp.choices[0].finish_reason
                }
            )
        except Exception as e:
            # Log error and return failure
            logging.error(f"OpenAI API error: {str(e)}")
            raise


class AnthropicProvider:
    """Anthropic API wrapper with enhanced features."""
    name = "anthropic"
    
    # Model pricing per 1K tokens (input, output)
    PRICING = {
        "claude-3-opus-20240229": (0.015, 0.075),
        "claude-3-sonnet-20240229": (0.003, 0.015),
        "claude-3-haiku-20240307": (0.00025, 0.00125),
        "claude-instant-1.2": (0.0008, 0.0024)
    }

    def __init__(
        self,
        credential_manager: Optional[Any] = None,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature
        self.anthropic = self._import_sdk()
        
        # Get API key from credential manager or environment
        if credential_manager is not None:
            self.api_key = credential_manager.get_api_key("anthropic")
        else:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            
        if not self.api_key:
            raise RuntimeError("Anthropic API key not available")

    @staticmethod
    def _import_sdk():
        try:
            return importlib.import_module("anthropic")
        except ImportError:
            raise RuntimeError("Anthropic SDK not installed. Install with: pip install anthropic")

    def count_tokens(self, text: str) -> int:
        """Estimate token count for Anthropic models."""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            return client.count_tokens(text)
        except (ImportError, AttributeError):
            # Fallback to word-based estimation
            return len(text.split()) + len(text) // 4

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage and model pricing."""
        model_pricing = self.PRICING.get(self.model, (0.003, 0.015))  # Default to Sonnet pricing
        input_cost, output_cost = model_pricing
        
        return (prompt_tokens * input_cost + completion_tokens * output_cost) / 1000

    def chat(self, role: str, content: str, **kwargs) -> LLMResponse:
        # Override model if specified in kwargs
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 1024)
        
        start = time.perf_counter()
        
        try:
            # Initialize client
            client = self.anthropic.Anthropic(api_key=self.api_key)
            
            # Count input tokens
            prompt_tokens = self.count_tokens(content)
            
            # Create message
            rsp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": role, "content": content}],
            )
            
            # Extract text from blocks
            text = "".join(block.text for block in rsp.content)
            completion_tokens = self.count_tokens(text)
            total_tokens = prompt_tokens + completion_tokens
            dur = (time.perf_counter() - start) * 1000
            
            cost = self.calculate_cost(prompt_tokens, completion_tokens)
            
            return LLMResponse(
                text=text,
                tokens=total_tokens,
                latency_ms=dur,
                cost_usd=cost,
                provider=self.name,
                model=model,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "stop_reason": rsp.stop_reason
                }
            )
        except Exception as e:
            # Log error and return failure
            logging.error(f"Anthropic API error: {str(e)}")
            raise


class LocalLLMProvider:
    """Provider for local LLM inference."""
    name = "local"
    
    def __init__(
        self,
        model_path: str = "models/codellama-13b",
        temperature: float = 0.1,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self._import_dependencies()
        self.model, self.tokenizer = self._load_model()

    def _import_dependencies(self):
        """Import required packages for local inference."""
        try:
            self.torch = importlib.import_module("torch")
            self.transformers = importlib.import_module("transformers")
        except ImportError:
            raise RuntimeError(
                "Dependencies for local LLM not installed. "
                "Install with: pip install torch transformers"
            )

    def _load_model(self):
        """Load the local model and tokenizer."""
        try:
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(self.model_path)
            model = self.transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch.float16,
                device_map="auto"
            )
            return model, tokenizer
        except Exception as e:
            logging.error(f"Error loading local model: {str(e)}")
            raise RuntimeError(f"Failed to load local model from {self.model_path}: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())

    def chat(self, role: str, content: str, **kwargs) -> LLMResponse:
        # Override temperature if specified in kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        
        start = time.perf_counter()
        
        try:
            # Format prompt based on role
            prompt = f"{role}: {content}\nAssistant: "
            
            # Tokenize input
            input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            prompt_token_count = input_tokens.shape[1]
            
            # Generate response
            with self.torch.no_grad():
                outputs = self.model.generate(
                    input_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            
            # Decode response (excluding prompt tokens)
            response_tokens = outputs[0, prompt_token_count:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Calculate metrics
            dur = (time.perf_counter() - start) * 1000
            completion_token_count = len(response_tokens)
            total_tokens = prompt_token_count + completion_token_count
            
            # Local models have no API cost
            cost = 0.0
            
            return LLMResponse(
                text=response_text,
                tokens=total_tokens,
                latency_ms=dur,
                cost_usd=cost,
                provider=self.name,
                model=self.model_path,
                metadata={
                    "prompt_tokens": prompt_token_count,
                    "completion_tokens": completion_token_count,
                    "local_inference": True
                }
            )
        except Exception as e:
            logging.error(f"Local LLM inference error: {str(e)}")
            raise


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Manager / Router
# ──────────────────────────────────────────────────────────────────────────────
class LLMManager:
    """
    Enhanced LLM Manager with fallback capabilities and secure credential handling.
    
    Features:
    - Multiple provider support with automatic fallback
    - Secure credential management
    - Integration with system configuration
    - Model selection based on task requirements
    - Retry logic for handling API errors
    
    Usage:
        lm = LLMManager(credential_manager=credential_manager)
        response = lm.chat(role="user", content="Question...")
    """

    def __init__(
        self, 
        credential_manager: Optional[Any] = None,
        config: Dict[str, Any] = None
    ) -> None:
        self.credential_manager = credential_manager
        self.config = config or {}
        
        # Extract configuration
        self.llm_config = self.config.get("llm", {})
        self.preferred_provider = self.llm_config.get("preferred", "auto")
        self.default_temperature = self.llm_config.get("temperature", 0.1)
        self.cost_budget_usd = self.llm_config.get("cost-budget-usd", 2.0)
        
        # Model configuration
        self.models = self.llm_config.get("models", {
            "primary": "gpt-4-turbo",
            "fallback": "claude-3-sonnet",
            "offline": "codellama-13b",
            "explanation": "gpt-4o-mini"
        })
        
        # Fallback and retry configuration
        self.fallback_strategy = self.llm_config.get("fallback-strategy", "sequential")
        self.retry_attempts = self.llm_config.get("retry-attempts", 3)
        
        # Initialize providers
        self.providers: Dict[str, BaseProvider] = {}
        self._init_providers()
        
        # Track usage for budget management
        self.total_cost_usd = 0.0
        self.last_reset_time = time.time()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ initialization
    def _init_providers(self) -> None:
        """Initialize providers based on configuration and available credentials."""
        # If a specific preferred provider is set (not "auto"), only initialize that one
        if self.preferred_provider != "auto":
            self._init_single_provider(self.preferred_provider)
        else:
            # Initialize all available providers
            self._init_all_providers()
            
        # Validate we have at least one provider
        if not self.providers:
            raise RuntimeError("No LLM providers could be initialized. Check credentials and dependencies.")
    
    def _init_single_provider(self, provider_name: str) -> None:
        """Initialize only the specified provider."""
        if provider_name == "openai":
            try:
                openai_provider = OpenAIProvider(
                    credential_manager=self.credential_manager, 
                    model=self.models.get("primary", "gpt-4-turbo"),
                    temperature=self.default_temperature
                )
                self.providers[openai_provider.name] = openai_provider
            except Exception as e:
                logging.warning(f"Failed to initialize OpenAI provider: {str(e)}")
        
        elif provider_name == "anthropic":
            try:
                anthropic_provider = AnthropicProvider(
                    credential_manager=self.credential_manager,
                    model=self.models.get("fallback", "claude-3-sonnet"),
                    temperature=self.default_temperature
                )
                self.providers[anthropic_provider.name] = anthropic_provider
            except Exception as e:
                logging.warning(f"Failed to initialize Anthropic provider: {str(e)}")
        
        elif provider_name == "local" and "offline" in self.models:
            try:
                local_provider = LocalLLMProvider(
                    model_path=self.models["offline"],
                    temperature=self.default_temperature
                )
                self.providers[local_provider.name] = local_provider
            except Exception as e:
                logging.warning(f"Failed to initialize Local provider: {str(e)}")
    
    def _init_all_providers(self) -> None:
        """Initialize all available providers with secure credentials."""
        # Initialize OpenAI if possible
        try:
            openai_provider = OpenAIProvider(
                credential_manager=self.credential_manager, 
                model=self.models.get("primary", "gpt-4-turbo"),
                temperature=self.default_temperature
            )
            self.providers[openai_provider.name] = openai_provider
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI provider: {str(e)}")
        
        # Initialize Anthropic if possible
        try:
            anthropic_provider = AnthropicProvider(
                credential_manager=self.credential_manager,
                model=self.models.get("fallback", "claude-3-sonnet"),
                temperature=self.default_temperature
            )
            self.providers[anthropic_provider.name] = anthropic_provider
        except Exception as e:
            logging.warning(f"Failed to initialize Anthropic provider: {str(e)}")
        
        # Initialize local provider if model path is configured
        if "offline" in self.models:
            try:
                local_provider = LocalLLMProvider(
                    model_path=self.models["offline"],
                    temperature=self.default_temperature
                )
                self.providers[local_provider.name] = local_provider
            except Exception as e:
                logging.warning(f"Failed to initialize Local provider: {str(e)}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ public API
    def set_preferred_provider(self, provider: str) -> None:
        """Set the preferred provider for subsequent calls."""
        if provider not in self.providers and provider != "auto":
            raise ValueError(f"Provider '{provider}' not available")
        self.preferred_provider = provider
    
    def chat(
        self,
        role: str,
        content: str,
        *,
        task_type: str = "general",
        complexity: str = "low",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retry_strategy: Union[RetryStrategy, str] = RetryStrategy.ALTERNATE_PROVIDER,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat message to the appropriate LLM provider.
        
        Args:
            role: Role of the message sender (e.g., "user", "assistant")
            content: Message content
            task_type: Type of task ("general", "explain", "code", etc.)
            complexity: Complexity level ("low", "medium", "high")
            provider: Specific provider to use, overrides automatic selection
            model: Specific model to use, overrides default model
            retry_strategy: Strategy for retrying on failure
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Standardized LLM response
        """
        # Check budget
        if self.total_cost_usd >= self.cost_budget_usd:
            # Reset if 24 hours have passed
            if time.time() - self.last_reset_time > 86400:  # 24 hours
                self.total_cost_usd = 0.0
                self.last_reset_time = time.time()
            else:
                raise RuntimeError(f"Cost budget exceeded: ${self.total_cost_usd:.2f} / ${self.cost_budget_usd:.2f}")
        
        # Convert string strategy to enum if needed
        if isinstance(retry_strategy, str):
            try:
                retry_strategy = RetryStrategy(retry_strategy)
            except ValueError:
                retry_strategy = RetryStrategy.ALTERNATE_PROVIDER
        
        # Select provider
        selected_provider = self._select_provider(provider, task_type, complexity)
        
        # Select model based on task if not explicitly provided
        if not model:
            model = self._select_model(task_type, complexity)
        
        # Add model to kwargs for provider
        kwargs["model"] = model
        
        # Track remaining providers for fallback
        available_providers = list(self.providers.keys())
        if selected_provider.name in available_providers:
            available_providers.remove(selected_provider.name)
        
        # Try primary provider
        for attempt in range(self.retry_attempts + 1):
            try:
                response = selected_provider.chat(role=role, content=content, **kwargs)
                
                # Update cost tracking
                self.total_cost_usd += response.cost_usd
                
                return response
            except Exception as e:
                logging.warning(f"Provider {selected_provider.name} failed on attempt {attempt+1}: {str(e)}")
                
                # Check if we should retry
                if attempt >= self.retry_attempts:
                    break
                
                # Determine next provider based on retry strategy
                if retry_strategy == RetryStrategy.NONE:
                    break
                elif retry_strategy == RetryStrategy.SAME_PROVIDER:
                    # Continue with same provider
                    continue
                elif retry_strategy in (RetryStrategy.ALTERNATE_PROVIDER, RetryStrategy.ALL_PROVIDERS):
                    # If we have alternative providers, try them
                    if available_providers:
                        next_provider_name = available_providers.pop(0)
                        selected_provider = self.providers[next_provider_name]
                        logging.info(f"Falling back to provider: {next_provider_name}")
                    elif retry_strategy == RetryStrategy.ALL_PROVIDERS:
                        # Reset available providers for ALL_PROVIDERS strategy
                        available_providers = [p for p in self.providers.keys() if p != selected_provider.name]
                        if available_providers:
                            next_provider_name = available_providers.pop(0)
                            selected_provider = self.providers[next_provider_name]
                            logging.info(f"Falling back to provider: {next_provider_name}")
                    else:
                        # No more providers to try
                        break
        
        # If all attempts failed, raise the last exception
        raise RuntimeError(f"All LLM providers failed after {self.retry_attempts+1} attempts")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ selection logic
    def _select_provider(
        self, 
        provider: Optional[str], 
        task_type: str, 
        complexity: str
    ) -> BaseProvider:
        """Select the appropriate provider based on constraints."""
        # Use explicit provider if specified and available
        if provider and provider in self.providers:
            return self.providers[provider]
        
        # Use preferred provider if specified and available
        if self.preferred_provider != "auto" and self.preferred_provider in self.providers:
            return self.providers[self.preferred_provider]
        
        # Task-based routing
        if task_type == "code" and complexity == "high" and "anthropic" in self.providers:
            return self.providers["anthropic"]  # Claude excels at code
        
        if task_type == "explain" and "openai" in self.providers:
            return self.providers["openai"]  # Good for explanations
        
        if complexity == "low" and "local" in self.providers:
            return self.providers["local"]  # Use local for simple tasks
        
        # Default to first available
        return next(iter(self.providers.values()))
    
    def _select_model(self, task_type: str, complexity: str) -> str:
        """Select appropriate model based on task requirements."""
        if task_type == "explain" or complexity == "low":
            return self.models.get("explanation", "gpt-4o-mini")
        
        if complexity == "high":
            return self.models.get("primary", "gpt-4-turbo")
        
        # Default to fallback model for medium complexity
        return self.models.get("fallback", "claude-3-sonnet")

    # ---------------------------------------------------------------- utility methods
    def available(self) -> List[str]:
        """Return list of available provider names."""
        return list(self.providers.keys())
    
    def get_budget_status(self) -> Dict[str, float]:
        """Return current budget status."""
        return {
            "total_cost_usd": self.total_cost_usd,
            "budget_usd": self.cost_budget_usd,
            "remaining_usd": max(0, self.cost_budget_usd - self.total_cost_usd),
            "percent_used": (self.total_cost_usd / self.cost_budget_usd) * 100 if self.cost_budget_usd > 0 else 100
        }
    
    def get_models(self) -> Dict[str, List[str]]:
        """Return available models grouped by provider."""
        models = {}
        for provider_name, provider in self.providers.items():
            if provider_name == "openai":
                models[provider_name] = ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
            elif provider_name == "anthropic":
                models[provider_name] = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            elif provider_name == "local":
                models[provider_name] = [provider.model_path]
        return models


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Token Counting Utilities
# ──────────────────────────────────────────────────────────────────────────────
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in a text string using the appropriate tokenizer.
    
    Args:
        text: The text to count tokens for
        model: The model whose tokenizer to use
        
    Returns:
        Estimated token count
    """
    # Try to use tiktoken for accurate counting
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            # Fallback to cl100k_base for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    except ImportError:
        # Fallback to rough estimate based on words and characters
        return len(text.split()) + len(text) // 4


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Simple demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize credential manager if available
    cred_manager = CredentialManager() if CredentialManager else None
    
    # Create LLM manager
    lm = LLMManager(credential_manager=cred_manager)
    
    print(f"Available providers: {lm.available()}")
    print(f"Available models: {lm.get_models()}")
    
    # Simple test
    try:
        response = lm.chat(
            role="user",
            content="Explain entropy in 3 sentences.",
            task_type="explain",
            complexity="low"
        )
        
        print(f"\nUsing {response.provider} ({response.model}):")
        print(f"Response: {response.text}")
        print(f"Tokens: {response.tokens}, Cost: ${response.cost_usd:.6f}")
        print(f"Budget status: {lm.get_budget_status()}")
    except Exception as e:
        print(f"Error: {str(e)}")
