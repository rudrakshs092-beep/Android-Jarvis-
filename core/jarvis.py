"""
core/jarvis.py — JARVIS Main Orchestrator (Bug-Fixed & Safe Memory Management)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from .brain import (
    AllProvidersFailedError,
    BrainError,
    Message,
    ModelResponse,
    ModelRouter,
    Role,
    build_router_from_config,
)

logger = logging.getLogger(__name__)


class JarvisState(Enum):
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    STOPPED = auto()


@dataclass
class ConversationTurn:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)
    provider: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class JarvisResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class JarvisConfig:
    providers: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str = (
        "You are JARVIS, an intelligent AI assistant running on Android. "
        "Be concise, helpful, and respectful of device resource constraints."
    )
    history_limit: int = 20
    strategy: str = "priority"
    max_tokens: int = 1024
    temperature: float = 0.7
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "JarvisConfig":
        provider_map = [
            ("gemini", "JARVIS_GEMINI_KEY"),
            ("claude", "JARVIS_CLAUDE_KEY"),
            ("openai", "JARVIS_OPENAI_KEY"),
            ("deepseek", "JARVIS_DEEPSEEK_KEY"),
            ("grok", "JARVIS_GROK_KEY"),
        ]
        providers: List[Dict[str, Any]] = []
        for idx, (name, env_var) in enumerate(provider_map):
            key = os.environ.get(env_var, "").strip()
            if key:
                providers.append({"name": name, "api_key": key, "priority": idx})

        return cls(
            providers=providers,
            strategy=os.environ.get("JARVIS_STRATEGY", "priority"),
            history_limit=int(os.environ.get("JARVIS_HISTORY_LIMIT", "20")),
            max_tokens=int(os.environ.get("JARVIS_MAX_TOKENS", "1024")),
            temperature=float(os.environ.get("JARVIS_TEMPERATURE", "0.7")),
            log_level=os.environ.get("JARVIS_LOG_LEVEL", "INFO"),
            system_prompt=os.environ.get(
                "JARVIS_SYSTEM_PROMPT",
                "You are JARVIS, an intelligent AI assistant running on Android. "
                "Be concise, helpful, and respectful of device resource constraints.",
            ),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JarvisConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


EventHandler = Callable[..., Any]


class EventBus:
    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = {}

    def on(self, event: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: EventHandler) -> None:
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: str, **kwargs: Any) -> None:
        for handler in list(self._handlers.get(event, [])):
            try:
                handler(**kwargs)
            except Exception as exc:
                logger.error("[EventBus] handler for '%s' raised: %s", event, exc)


class ConversationStore:
    def __init__(self, limit: int = 20) -> None:
        self._limit = max(2, limit)
        self._turns: List[ConversationTurn] = []

    def add(self, turn: ConversationTurn) -> None:
        self._turns.append(turn)
        while len(self._turns) > self._limit:
            self._turns.pop(0)

    def pop_last(self) -> None:
        """Remove the last unhandled/failed turn without wiping history."""
        if self._turns:
            self._turns.pop()

    def to_messages(self) -> List[Message]:
        return [Message(role=t.role, content=t.content) for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()

    def snapshot(self) -> List[ConversationTurn]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)


class Jarvis:
    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        *,
        router: Optional[ModelRouter] = None,
    ) -> None:
        self._config = config or JarvisConfig.from_env()
        self._router: Optional[ModelRouter] = router
        self._state = JarvisState.UNINITIALIZED
        self._history = ConversationStore(self._config.history_limit)
        self._events = EventBus()
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._state in (JarvisState.RUNNING, JarvisState.INITIALIZING):
                return

            self._state = JarvisState.INITIALIZING
            self._events.emit("initializing")

            try:
                if self._router is None:
                    if not self._config.providers:
                        raise BrainError("No AI providers configured.")
                    self._router = build_router_from_config(
                        {
                            "strategy": self._config.strategy,
                            "providers": self._config.providers,
                        }
                    )

                self._state = JarvisState.RUNNING
                self._events.emit("ready", providers=self._router.provider_names)

            except Exception as exc:
                self._state = JarvisState.UNINITIALIZED
                self._events.emit("init_failed", error=str(exc))
                raise

    async def shutdown(self) -> None:
        if self._state == JarvisState.STOPPED:
            return
        if self._router:
            await self._router.close()
        self._state = JarvisState.STOPPED
        self._events.emit("stopped")

    async def __aenter__(self) -> "Jarvis":
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    async def chat(
        self,
        user_input: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        include_history: bool = True,
    ) -> JarvisResponse:
        if self._state != JarvisState.RUNNING:
            await self.initialize()

        user_input = user_input.strip()
        if not user_input:
            return JarvisResponse(text="", provider="none", model="none", latency_ms=0.0, error="Empty input.")

        user_turn = ConversationTurn(role=Role.USER, content=user_input)
        self._history.add(user_turn)

        messages = self._history.to_messages() if include_history else [
            Message(role=Role.USER, content=user_input)
        ]

        self._events.emit("request", user_input=user_input)

        try:
            assert self._router is not None
            model_response: ModelResponse = await self._router.complete(
                messages=messages,
                system_prompt=system_prompt or self._config.system_prompt,
                max_tokens=max_tokens or self._config.max_tokens,
                temperature=temperature if temperature is not None else self._config.temperature,
            )

            assistant_turn = ConversationTurn(
                role=Role.ASSISTANT,
                content=model_response.text,
                provider=model_response.provider,
                latency_ms=model_response.latency_ms,
            )
            self._history.add(assistant_turn)
            self._events.emit("response", response=model_response)

            return JarvisResponse(
                text=model_response.text,
                provider=model_response.provider,
                model=model_response.model,
                latency_ms=model_response.latency_ms,
                prompt_tokens=model_response.prompt_tokens,
                completion_tokens=model_response.completion_tokens,
            )

        except AllProvidersFailedError as exc:
            self._history.pop_last()  # Safely pop only the failed turn
            self._events.emit("error", error=str(exc))
            return JarvisResponse(
                text="I'm unable to reach any AI provider right now.",
                provider="none",
                model="none",
                latency_ms=0.0,
                error=str(exc),
            )

        except Exception as exc:
            self._history.pop_last()
            self._events.emit("error", error=str(exc))
            return JarvisResponse(
                text="An unexpected error occurred.",
                provider="none",
                model="none",
                latency_ms=0.0,
                error=str(exc),
            )

    def clear_history(self) -> None:
        self._history.clear()

    def get_history(self) -> List[ConversationTurn]:
        return self._history.snapshot()

    def on(self, event: str, handler: EventHandler) -> None:
        self._events.on(event, handler)

    def off(self, event: str, handler: EventHandler) -> None:
        self._events.off(event, handler)
