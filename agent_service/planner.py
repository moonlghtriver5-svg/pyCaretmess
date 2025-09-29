"""Simple planner that decides which tool or response strategy to execute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class SupportsMessage(Protocol):
    """Protocol representing minimal chat message attributes."""

    role: str
    content: str


@dataclass(slots=True)
class PlannerDecision:
    """Represents the planner decision and optional metadata."""

    action: str
    rationale: str


class Planner:
    """Rule based planner inspired by agent orchestration frameworks."""

    def decide(self, messages: Iterable[SupportsMessage]) -> PlannerDecision:
        """Decide the next action based on the latest user message."""

        try:
            last_message = next(
                message for message in reversed(list(messages)) if message.role == "user"
            )
        except StopIteration:
            return PlannerDecision(
                action="plain_chat", rationale="No user message provided; defaulting to chat."
            )

        text = last_message.content.lower()
        if any(keyword in text for keyword in {"sweetviz", "ydata", "eda"}):
            return PlannerDecision(
                action="generate_eda",
                rationale="Detected request for automated EDA tooling (Sweetviz/YData).",
            )

        if "pycaret" in text or "train" in text:
            return PlannerDecision(
                action="run_pycaret",
                rationale="Detected modeling intent referencing PyCaret or training instructions.",
            )

        return PlannerDecision(action="plain_chat", rationale="Defaulting to LLM chat response.")


__all__ = ["Planner", "PlannerDecision"]
