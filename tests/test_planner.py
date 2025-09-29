"""Unit tests for the planner module."""

from dataclasses import dataclass

from datetime import UTC, datetime

from datetime import datetime

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_service.planner import Planner


@dataclass
class DummyMessage:
    role: str
    content: str
    timestamp: datetime


def _message(content: str) -> DummyMessage:
    return DummyMessage(
        role="user", content=content, timestamp=datetime.now(UTC)
    )


def _message(content: str) -> DummyMessage:
    return DummyMessage(role="user", content=content, timestamp=datetime.utcnow())



def test_detects_eda_keywords():
    planner = Planner()
    decision = planner.decide([_message("please run sweetviz")])
    assert decision.action == "generate_eda"
    assert "EDA" in decision.rationale


def test_detects_pycaret_requests():
    planner = Planner()
    decision = planner.decide([_message("train a pycaret model")])
    assert decision.action == "run_pycaret"


def test_defaults_to_plain_chat():
    planner = Planner()
    decision = planner.decide([_message("hello there")])
    assert decision.action == "plain_chat"
