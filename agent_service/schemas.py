"""Pydantic schemas shared across FastAPI routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request payload for /chat endpoint."""

    conversation_id: str | None = Field(
        default=None, description="Client supplied identifier for the conversation."
    )
    messages: list[ChatMessage]
    dataset_path: str | None = Field(
        default=None,
        description="Optional relative path to a dataset that the agent can access for EDA.",
    )
    target: str | None = Field(
        default=None, description="Optional target column used for modeling tasks."
    )


class ChatResponse(BaseModel):
    """Response payload returned from /chat endpoint."""

    conversation_id: str
    reply: str
    planner_action: str
    artifacts: dict[str, Any] | None = None


class ActionRequest(BaseModel):
    """Request payload for direct tool execution."""

    action: Literal["generate_eda", "run_pycaret"]
    dataset_path: str
    target: str | None = None


class ActionResponse(BaseModel):
    """Response returned after executing a tool directly."""

    action: str
    detail: str
    artifacts: dict[str, Any] | None = None


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ActionRequest",
    "ActionResponse",
]
