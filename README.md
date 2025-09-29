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


## Running in GitHub Codespaces

The repository works well inside a GitHub Codespace. After creating a Codespace for
the repo, open a terminal and run:

1. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run automated checks:**

   ```bash
   pytest
   ```

4. **Launch the backend API (exposes port 8000):**

   ```bash
   uvicorn agent_service.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   When prompted, allow the Codespace to make the forwarded port public or private as
   needed.

5. **Launch the Streamlit frontend in a second terminal (port 8501):**

   ```bash
   streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```

GitHub Codespaces automatically forwards both ports; open the generated URLs from the
Ports panel to interact with the API or the UI. Environment variables such as
`OPENROUTER_API_KEY` can be added via the Codespace “Secrets” settings.

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
