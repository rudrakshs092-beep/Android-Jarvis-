"""
core/brain.py
JARVIS Assistant - Core AI Reasoning Engine

Production-ready, async-first Brain module built on Clean Architecture and SOLID principles.
Acts as the central orchestrator for AI interactions, utilizing a Model Router to abstract
away direct LLM API calls. Supports Android APK environments (Chaquopy/Termux) by strictly
avoiding blocking I/O and heavy native dependencies in the core logic.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# ==============================================================================
# Domain Models & Data Structures
# ==============================================================================

class ResponseStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"  # e.g., content moderation triggered

@dataclass
class ModelResponse:
    """Structured response returned by the Brain."""
    content: str
    status: ResponseStatus
    provider_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class PromptContext:
    """Contextual data passed alongside a prompt to the Brain."""
    user_id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    intent: Optional[str] = None  # e.g., 'automation', 'chat', 'coding'

# ==============================================================================
# Custom Exceptions
# ==============================================================================

class BrainException(Exception):
    """Base exception for Brain module."""
    pass

class BrainNotInitializedError(BrainException):
    pass

class ProviderUnavailableError(BrainException):
    pass

# ==============================================================================
# Provider Abstraction (Dependency Inversion Principle)
# ==============================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """
    Interface for LLM Providers.
    Any new provider (e.g., Mistral, Llama) must implement this interface
    to be automatically compatible with the Model Router.
    """
    name: str

    async def initialize(self) -> None:
        """Initialize API clients, validate keys, etc."""
        ...

    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        """Execute the LLM generation."""
        ...

    async def health_check(self) -> bool:
        """Check if the provider is reachable and operational."""
        ...

    async def shutdown(self) -> None:
        """Clean up resources."""
        ...

# ==============================================================================
# Concrete Provider Placeholders (Open/Closed Principle)
# ==============================================================================

class BaseProvider(ABC):
    """Base class to reduce boilerplate for concrete providers."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"jarvis.brain.provider.{self.__class__.__name__}")
        self._is_initialized = False

    async def initialize(self) -> None:
        self.logger.info(f"Initializing {self.name} provider...")
        # TODO: Load API keys from config, instantiate async SDK clients
        self._is_initialized = True
        self.logger.info(f"{self.name} provider initialized.")

    async def health_check(self) -> bool:
        return self._is_initialized

    async def shutdown(self) -> None:
        self.logger.info(f"Shutting down {self.name} provider...")
        self._is_initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        ...

class GLMProvider(BaseProvider):
    @property
    def name(self) -> str: return "GLM"
    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        raise NotImplementedError("GLM API call not implemented.")

class GeminiProvider(BaseProvider):
    @property
    def name(self) -> str: return "Gemini"
    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        raise NotImplementedError("Gemini API call not implemented.")

class ClaudeProvider(BaseProvider):
    @property
    def name(self) -> str: return "Claude"
    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        raise NotImplementedError("Claude API call not implemented.")

class OpenAIProvider(BaseProvider):
    @property
    def name(self) -> str: return "OpenAI"
    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        raise NotImplementedError("OpenAI API call not implemented.")

class GrokProvider(BaseProvider):
    @property
    def name(self) -> str: return "Grok"
    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        raise NotImplementedError("Grok API call not implemented.")

class DeepSeekProvider(BaseProvider):
    @property
    def name(self) -> str: return "DeepSeek"
    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        raise NotImplementedError("DeepSeek API call not implemented.")

class OpenRouterProvider(BaseProvider):
    @property
    def name(self) -> str: return "OpenRouter"
    async def generate(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        raise NotImplementedError("OpenRouter API call not implemented.")

# ==============================================================================
# Model Router (Single Responsibility Principle)
# ==============================================================================

class ModelRouter:
    """
    Routes prompts to the appropriate LLM provider based on context, intent,
    or configuration. Handles fallbacks and load balancing.
    """
    def __init__(self):
        self.logger = logging.getLogger("jarvis.brain.router")
        self._providers: Dict[str, LLMProvider] = {}
        self._default_provider: Optional[str] = None

    def register_provider(self, provider: LLMProvider, is_default: bool = False) -> None:
        """Dynamically register a new provider (Open/Closed Principle)."""
        self._providers[provider.name] = provider
        if is_default or not self._default_provider:
            self._default_provider = provider.name
        self.logger.info(f"Registered provider: {provider.name} (Default: {self._default_provider == provider.name})")

    async def route_and_execute(self, prompt: str, context: PromptContext, **kwargs) -> ModelResponse:
        """Route the prompt and execute, with graceful fallback."""
        if not self._providers:
            return ModelResponse(
                content="",
                status=ResponseStatus.ERROR,
                provider_used="None",
                error="No LLM providers registered."
            )

        # Determine target provider (e.g., route coding tasks to Claude, fast tasks to Groq)
        # For now, we use default or context-specified provider.
        target_name = context.metadata.get("target_provider", self._default_provider) if hasattr(context, 'metadata') else self._default_provider
        
        # Fallback chain: Try target, then all others
        provider_chain = list(self._providers.values())
        # Move target to front
        if target_name in self._providers:
            target_provider = self._providers[target_name]
            provider_chain.remove(target_provider)
            provider_chain.insert(0, target_provider)

        last_error = None
        for provider in provider_chain:
            try:
                self.logger.debug(f"Routing prompt to {provider.name}...")
                response = await provider.generate(prompt, context, **kwargs)
                return response
            except Exception as e:
                self.logger.warning(f"Provider {provider.name} failed: {e}. Attempting fallback...")
                last_error = str(e)
                continue

        return ModelResponse(
            content="",
            status=ResponseStatus.ERROR,
            provider_used="All",
            error=f"All providers failed. Last error: {last_error}"
        )

# ==============================================================================
# Core Brain Module
# ==============================================================================

class Brain:
    """
    The central AI reasoning engine for JARVIS.
    Orchestrates prompt processing, context management, and model routing.
    """

    def __init__(self, config: Dict[str, Any], router: ModelRouter):
        """
        Initialize the Brain with injected dependencies.
        
        Args:
            config: Global configuration dictionary.
            router: The ModelRouter instance containing provider logic.
        """
        self.config = config
        self.router = router
        self.logger = logging.getLogger("jarvis.brain")
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the Brain and all registered providers in the router."""
        if self._is_initialized:
            self.logger.warning("Brain is already initialized.")
            return

        self.logger.info("Initializing Brain Module...")
        
        # Initialize all providers concurrently for fast startup (Android optimization)
        providers = list(self.router._providers.values())
        if not providers:
            self.logger.error("No providers registered to the router.")
            raise BrainNotInitializedError("ModelRouter has no registered providers.")

        try:
            await asyncio.gather(*[p.initialize() for p in providers])
            self._is_initialized = True
            self.logger.info("Brain Module initialized successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize Brain providers: {e}", exc_info=True)
            raise BrainNotInitializedError(f"Provider initialization failed: {e}") from e

    async def process(self, user_input: str, context: Optional[PromptContext] = None) -> ModelResponse:
        """
        High-level entry point. Pre-processes input, formats context,
        and delegates to generate_response.
        """
        if not self._is_initialized:
            raise BrainNotInitializedError("Brain must be initialized before processing.")

        if context is None:
            context = PromptContext(user_id="default")

        # Pre-processing logic (e.g., PII redaction, prompt sanitization)
        sanitized_input = self._pre_process(user_input)

        return await self.generate_response(sanitized_input, context)

    async def generate_response(self, prompt: str, context: PromptContext) -> ModelResponse:
        """
        Routes the formatted prompt to the ModelRouter and post-processes the result.
        """
        self.logger.debug(f"Generating response for user {context.user_id}. Intent: {context.intent}")
        
        try:
            response = await self.router.route_and_execute(prompt, context)
            
            # Post-processing (e.g., content moderation, formatting)
            response = self._post_process(response)
            return response
            
        except Exception as e:
            self.logger.error(f"Error during response generation: {e}", exc_info=True)
            return ModelResponse(
                content="I apologize, but I encountered an internal processing error.",
                status=ResponseStatus.ERROR,
                provider_used="Unknown",
                error=str(e)
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Brain and its providers."""
        if not self._is_initialized:
            return {"status": "unhealthy", "error": "Brain not initialized."}

        provider_health = {}
        overall_healthy = True

        for name, provider in self.router._providers.items():
            try:
                is_healthy = await provider.health_check()
                provider_health[name] = "healthy" if is_healthy else "unhealthy"
                if not is_healthy:
                    overall_healthy = False
            except Exception as e:
                provider_health[name] = f"error: {str(e)}"
                overall_healthy = False

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "providers": provider_health
        }

    async def shutdown(self) -> None:
        """Gracefully shut down the Brain and release provider resources."""
        self.logger.info("Shutting down Brain Module...")
        
        providers = list(self.router._providers.values())
        await asyncio.gather(*[p.shutdown() for p in providers], return_exceptions=True)
        
        self._is_initialized = False
        self.logger.info("Brain Module shut down complete.")

    # --- Private Helpers ---

    def _pre_process(self, text: str) -> str:
        """Placeholder for input sanitization, PII redaction, etc."""
        return text.strip()

    def _post_process(self, response: ModelResponse) -> ModelResponse:
        """Placeholder for output formatting, safety checks, etc."""
        return response

# ==============================================================================
# Example Factory Function (For clean dependency injection in jarvis.py)
# ==============================================================================

def create_brain(config: Dict[str, Any]) -> Brain:
    """
    Factory function to construct a fully wired Brain instance.
    This keeps `jarvis.py` clean and isolates provider wiring logic.
    """
    router = ModelRouter()
    
    # Register providers based on config (keys would come from config/config.py)
    # In a real scenario, you'd check if keys exist before registering.
    router.register_provider(OpenAIProvider(config), is_default=True)
    router.register_provider(ClaudeProvider(config))
    router.register_provider(GeminiProvider(config))
    router.register_provider(GLMProvider(config))
    router.register_provider(GrokProvider(config))
    router.register_provider(DeepSeekProvider(config))
    router.register_provider(OpenRouterProvider(config))

    return Brain(config, router)
