"""
core/brain.py — JARVIS AI Brain (Bug-Fixed & Android-Optimized)
Dynamic multi-provider API router with async fallback pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass
class ModelResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def sanitize(self) -> "ModelResponse":
        """Return a copy with the text stripped of leading/trailing whitespace."""
        return ModelResponse(
            text=self.text.strip(),
            provider=self.provider,
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_ms=self.latency_ms,
            raw=self.raw,
        )


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    model: str
    base_url: str
    timeout: float = 30.0
    max_retries: int = 2
    priority: int = 0          # lower = higher priority
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class BrainError(Exception):
    """Base exception for brain module."""


class AllProvidersFailedError(BrainError):
    """Raised when every provider in the fallback chain has failed."""

    def __init__(self, errors: Dict[str, str]) -> None:
        self.errors = errors
        msgs = "; ".join(f"{k}: {v}" for k, v in errors.items())
        super().__init__(f"All providers failed — {msgs}")


# ---------------------------------------------------------------------------
# Abstract Base Provider
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    """Interface every AI provider adapter must implement."""

    def __init__(self, config: ProviderConfig) -> None:
        self._cfg = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._cfg.timeout)
            # Mobile-optimized TCP Connector settings
            connector = aiohttp.TCPConnector(limit=4, force_close=True, enable_cleanup_closed=True)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self._auth_headers(),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @abstractmethod
    def _auth_headers(self) -> Dict[str, str]:
        """Return provider-specific authentication headers."""

    @abstractmethod
    def _build_payload(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Serialize messages into the provider's request body."""

    @abstractmethod
    def _parse_response(self, data: Dict[str, Any]) -> ModelResponse:
        """Deserialize the provider's raw JSON response into a ModelResponse."""

    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        payload = self._build_payload(messages, system_prompt, **kwargs)
        session = await self._get_session()

        for attempt in range(1, self._cfg.max_retries + 2):
            t0 = time.perf_counter()
            try:
                async with session.post(self._cfg.base_url, json=payload) as resp:
                    body = await resp.json(content_type=None)
                    if resp.status >= 400:
                        err = body.get("error", {})
                        msg = err.get("message", str(body)) if isinstance(err, dict) else str(err)
                        raise BrainError(f"HTTP {resp.status}: {msg}")
                    response = self._parse_response(body)
                    response.latency_ms = (time.perf_counter() - t0) * 1000
                    return response.sanitize()

            except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError) as exc:
                if attempt > self._cfg.max_retries:
                    raise BrainError(f"Connection failed after {attempt} attempts: {exc}") from exc
                wait = 2 ** (attempt - 1)
                logger.warning("[%s] attempt %d failed (%s), retrying in %ds", self._cfg.name, attempt, exc, wait)
                await asyncio.sleep(wait)

            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as exc:
                raise BrainError(f"Request timed out after {self._cfg.timeout}s") from exc

        raise BrainError("Exhausted retry budget unexpectedly")

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def priority(self) -> int:
        return self._cfg.priority


# ---------------------------------------------------------------------------
# Provider Implementations
# ---------------------------------------------------------------------------

class GeminiProvider(BaseProvider):
    """Google Gemini (generateContent REST API)."""

    def _auth_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _build_payload(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = "user" if msg.role == Role.USER else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        payload: Dict[str, Any] = {"contents": contents}
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        payload["generationConfig"] = {
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 1024),
        }
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> ModelResponse:
        try:
            candidate = data["candidates"][0]
            text = candidate["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return ModelResponse(
                text=text,
                provider=self._cfg.name,
                model=self._cfg.model,
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
                raw=data,
            )
        except (KeyError, IndexError) as exc:
            raise BrainError(f"Gemini parse error: {exc} — raw={json.dumps(data)[:200]}") from exc

    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        # Fixed URL Param formatting
        url = self._cfg.base_url
        if "?key=" not in url:
            url = f"{url}?key={self._cfg.api_key}"
            
        payload = self._build_payload(messages, system_prompt, **kwargs)
        session = await self._get_session()

        for attempt in range(1, self._cfg.max_retries + 2):
            t0 = time.perf_counter()
            try:
                async with session.post(url, json=payload) as resp:
                    body = await resp.json(content_type=None)
                    if resp.status >= 400:
                        raise BrainError(f"HTTP {resp.status}: {body}")
                    response = self._parse_response(body)
                    response.latency_ms = (time.perf_counter() - t0) * 1000
                    return response.sanitize()
            except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError) as exc:
                if attempt > self._cfg.max_retries:
                    raise BrainError(str(exc)) from exc
                await asyncio.sleep(2 ** (attempt - 1))
            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as exc:
                raise BrainError(f"Timeout: {exc}") from exc

        raise BrainError("Retry budget exhausted")


class ClaudeProvider(BaseProvider):
    """Anthropic Claude (Messages API)."""

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._cfg.api_key,
            "anthropic-version": self._cfg.extra.get("api_version", "2023-06-01"),
            "content-type": "application/json",
        }

    def _build_payload(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._cfg.model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [m.to_dict() for m in messages if m.role != Role.SYSTEM],
        }
        if system_prompt:
            payload["system"] = system_prompt
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> ModelResponse:
        try:
            text = data["content"][0]["text"]
            usage = data.get("usage", {})
            return ModelResponse(
                text=text,
                provider=self._cfg.name,
                model=data.get("model", self._cfg.model),
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                raw=data,
            )
        except (KeyError, IndexError) as exc:
            raise BrainError(f"Claude parse error: {exc}") from exc


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible Chat Completions."""

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(m.to_dict() for m in messages)

        payload: Dict[str, Any] = {
            "model": self._cfg.model,
            "messages": msgs,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
        }
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> ModelResponse:
        try:
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return ModelResponse(
                text=text,
                provider=self._cfg.name,
                model=data.get("model", self._cfg.model),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                raw=data,
            )
        except (KeyError, IndexError) as exc:
            raise BrainError(f"OpenAI-compat parse error: {exc}") from exc


_PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "model": "gemini-1.5-flash",
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-haiku-20240307",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
    },
    "grok": {
        "base_url": "https://api.x.ai/v1/chat/completions",
        "model": "grok-beta",
    },
}


def build_provider(name: str, api_key: str, **overrides: Any) -> BaseProvider:
    name_lower = name.lower()
    defaults = _PROVIDER_DEFAULTS.get(name_lower)
    if defaults is None:
        raise ValueError(f"Unknown provider '{name}'. Known: {list(_PROVIDER_DEFAULTS)}")

    base_url = overrides.pop("base_url", defaults["base_url"])
    model = overrides.pop("model", defaults["model"])

    if name_lower == "gemini":
        base_url = base_url.format(model=model)

    cfg = ProviderConfig(
        name=name_lower,
        api_key=api_key,
        model=model,
        base_url=base_url,
        **overrides,
    )

    if name_lower == "gemini":
        return GeminiProvider(cfg)
    if name_lower == "claude":
        return ClaudeProvider(cfg)
    return OpenAIProvider(cfg)


class RouterStrategy(Enum):
    PRIORITY = auto()
    ROUND_ROBIN = auto()
    FASTEST = auto()


class ModelRouter:
    def __init__(
        self,
        providers: Sequence[BaseProvider],
        strategy: RouterStrategy = RouterStrategy.PRIORITY,
    ) -> None:
        if not providers:
            raise ValueError("ModelRouter requires at least one provider.")
        self._providers: List[BaseProvider] = sorted(
            [p for p in providers if p.enabled],
            key=lambda p: p.priority,
        )
        self._strategy = strategy
        self._rr_index = 0
        self._failure_counts: Dict[str, int] = {p.name: 0 for p in self._providers}

    async def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if self._strategy == RouterStrategy.FASTEST:
            return await self._race(messages, system_prompt, **kwargs)
        if self._strategy == RouterStrategy.ROUND_ROBIN:
            return await self._round_robin(messages, system_prompt, **kwargs)
        return await self._priority_fallback(messages, system_prompt, **kwargs)

    async def close(self) -> None:
        await asyncio.gather(*(p.close() for p in self._providers), return_exceptions=True)

    @property
    def provider_names(self) -> List[str]:
        return [p.name for p in self._providers]

    async def _priority_fallback(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> ModelResponse:
        errors: Dict[str, str] = {}
        for provider in self._providers:
            try:
                response = await provider.complete(messages, system_prompt, **kwargs)
                self._failure_counts[provider.name] = 0
                return response
            except Exception as exc:
                self._failure_counts[provider.name] += 1
                errors[provider.name] = str(exc)

        raise AllProvidersFailedError(errors)

    async def _round_robin(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> ModelResponse:
        count = len(self._providers)
        errors: Dict[str, str] = {}
        for _ in range(count):
            provider = self._providers[self._rr_index % count]
            self._rr_index = (self._rr_index + 1) % count
            try:
                return await provider.complete(messages, system_prompt, **kwargs)
            except Exception as exc:
                errors[provider.name] = str(exc)
        raise AllProvidersFailedError(errors)

    async def _race(
        self,
        messages: List[Message],
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> ModelResponse:
        loop = asyncio.get_event_loop()
        tasks: Dict[asyncio.Task, str] = {}
        for provider in self._providers:
            task = loop.create_task(provider.complete(messages, system_prompt, **kwargs))
            tasks[task] = provider.name

        errors: Dict[str, str] = {}
        pending: set = set(tasks.keys())

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is None:
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    return task.result()
                errors[tasks[task]] = str(exc)

        raise AllProvidersFailedError(errors)

    def failure_counts(self) -> Dict[str, int]:
        return dict(self._failure_counts)


def build_router_from_config(config: Dict[str, Any]) -> ModelRouter:
    strategy_map = {
        "priority": RouterStrategy.PRIORITY,
        "round_robin": RouterStrategy.ROUND_ROBIN,
        "fastest": RouterStrategy.FASTEST,
    }
    strategy = strategy_map.get(config.get("strategy", "priority"), RouterStrategy.PRIORITY)

    providers: List[BaseProvider] = []
    for entry in config.get("providers", []):
        entry = dict(entry)
        name = entry.pop("name")
        api_key = entry.pop("api_key", "")
        if not api_key:
            continue
        try:
            providers.append(build_provider(name, api_key, **entry))
        except ValueError as exc:
            logger.error("Skipping provider '%s': %s", name, exc)

    if not providers:
        raise BrainError("No valid providers found in config.")

    return ModelRouter(providers, strategy=strategy)
