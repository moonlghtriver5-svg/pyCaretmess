# PyCaret Copilot Stack

A reference implementation of a PyCaret-focused agent that combines automated EDA,
rule-based planner logic, and a conversational interface. The repository contains:

- **FastAPI backend** (`agent_service/`) with endpoints for chat, automated EDA, and
  PyCaret experiment execution.
- **Tooling layer** that generates Sweetviz and YData-Profiling reports and derives
  PyCaret `setup()` recommendations from dataset heuristics.
- **Streamlit frontend** (`frontend/`) that offers a lightweight chat-like interface for
  uploading datasets and triggering backend actions.
- **Unit tests** validating planner routing decisions.

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Export your OpenRouter API key (needed only for live LLM chat):

   ```bash
   export OPENROUTER_API_KEY="sk-..."
   ```

3. Launch the backend:

   ```bash
   uvicorn agent_service.main:app --reload
   ```

4. In a separate terminal start the Streamlit UI:

   ```bash
   streamlit run frontend/streamlit_app.py
   ```

Upload a dataset through the UI to generate automated EDA reports or trigger PyCaret
model training. For scripted usage you can POST directly to `/chat` or `/actions`.

## Testing

Run the unit tests with:

```bash
pytest
```

## Project layout

```
agent_service/
  config.py          # Environment-driven configuration
  llm_client.py      # OpenRouter HTTP client wrapper
  main.py            # FastAPI entrypoint
  planner.py         # Rule-based decision maker
  schemas.py         # Pydantic models
  tools.py           # EDA + PyCaret orchestration helpers
frontend/
  streamlit_app.py   # Lightweight chat UI
requirements.txt      # Python dependencies
```

Automated EDA artifacts are written to the `artifacts/` directory by default.
