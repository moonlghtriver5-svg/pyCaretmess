"""HTTP client for interacting with OpenRouter compatible language models."""

from __future__ import annotations

import logging
from typing import Iterable

import requests

from .config import get_settings
from .schemas import ChatMessage

LOGGER = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when an upstream LLM provider returns an error."""


class LLMClient:
    """Thin wrapper around the OpenRouter chat completions API."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _format_messages(self, messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    def chat(self, messages: Iterable[ChatMessage]) -> str:
        """Send a chat completion request to the configured model."""

        if not self.settings.openrouter_api_key:
            raise LLMClientError(
                "OPENROUTER_API_KEY is not configured; unable to reach the LLM provider."
            )

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "HTTP-Referer": "https://example.com",
            "X-Title": "PyCaret Copilot",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": self._format_messages(messages),
        }

        response = requests.post(
            f"{self.settings.openrouter_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        if response.status_code != 200:
            LOGGER.error("LLM provider error %s: %s", response.status_code, response.text)
            raise LLMClientError(
                f"OpenRouter request failed with {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMClientError(
                "Unexpected response format received from OpenRouter."
            ) from exc


__all__ = ["LLMClient", "LLMClientError"]
