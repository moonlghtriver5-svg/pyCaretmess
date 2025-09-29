"""FastAPI entrypoint exposing chat and action endpoints."""

from __future__ import annotations

import logging
import uuid


from fastapi import FastAPI, HTTPException

from .config import get_settings
from .llm_client import LLMClient, LLMClientError
from .planner import Planner
from .schemas import ActionRequest, ActionResponse, ChatRequest, ChatResponse
from .tools import (
    load_dataset,
    profile_dataset,
    recommend_pycaret_setup,
    run_pycaret_experiment,
    summarize_recommendation,
)

LOGGER = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Construct the FastAPI application."""

    settings = get_settings()
    planner = Planner()
    llm_client = LLMClient()

    app = FastAPI(title="PyCaret Copilot Agent", version="0.1.0")

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": settings.llm_model}

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: ChatRequest) -> ChatResponse:
        decision = planner.decide(request.messages)
        conversation_id = request.conversation_id or str(uuid.uuid4())

        if decision.action == "generate_eda":
            if not request.dataset_path:
                raise HTTPException(
                    status_code=400, detail="dataset_path is required for EDA generation."
                )
            artifacts = profile_dataset(request.dataset_path, request.target)
            reply = (
                "Generated automated EDA artifacts. "
                "Sweetviz and YData-Profiling reports are ready for review."
            )
            return ChatResponse(
                conversation_id=conversation_id,
                reply=reply,
                planner_action=decision.action,
                artifacts=artifacts,
            )

        if decision.action == "run_pycaret":
            if not request.dataset_path or not request.target:
                raise HTTPException(
                    status_code=400,
                    detail="dataset_path and target are required to launch PyCaret experiments.",
                )
            dataframe = load_dataset(request.dataset_path)
            recommendation = recommend_pycaret_setup(dataframe, request.target)
            summary = summarize_recommendation(recommendation)
            reply = (
                "Prepared PyCaret setup recommendations based on dataset heuristics. "
                "You can execute the experiment via the /actions endpoint."
            )
            return ChatResponse(
                conversation_id=conversation_id,
                reply=f"{reply}\n\n{summary}",
                planner_action=decision.action,
                artifacts={"recommendation": recommendation},
            )

        try:
            response_text = llm_client.chat(request.messages)
        except LLMClientError as exc:  # pragma: no cover - requires API key
            LOGGER.exception("LLM provider error")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return ChatResponse(
            conversation_id=conversation_id,
            reply=response_text,
            planner_action=decision.action,
        )

    @app.post("/actions", response_model=ActionResponse, tags=["actions"])
    async def actions(request: ActionRequest) -> ActionResponse:
        if request.action == "generate_eda":
            artifacts = profile_dataset(request.dataset_path, request.target)
            return ActionResponse(
                action=request.action,
                detail="Generated Sweetviz and YData-Profiling reports.",
                artifacts=artifacts,
            )

        if request.action == "run_pycaret":
            if not request.target:
                raise HTTPException(status_code=400, detail="target is required for PyCaret runs.")
            result = run_pycaret_experiment(request.dataset_path, request.target)
            detail = (
                "PyCaret experiment finished. Model saved to {model_path} and leaderboard at {leaderboard}."
            ).format(
                model_path=result["model_path"], leaderboard=result["leaderboard_path"]
            )
            return ActionResponse(action=request.action, detail=detail, artifacts=result)

        raise HTTPException(status_code=400, detail=f"Unknown action {request.action}")

    return app


app = create_app()
