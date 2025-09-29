# PyCaret Copilot Frontend

This directory contains a lightweight Streamlit interface that talks to the FastAPI
backend. It provides a minimal chat-like workflow for uploading datasets, triggering
automated EDA, and running PyCaret experiments.

## Running locally

1. Install Python dependencies from the project root:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend API (from the repository root):

   ```bash
   uvicorn agent_service.main:app --reload
   ```

3. In a separate terminal, launch Streamlit:

   ```bash
   streamlit run frontend/streamlit_app.py
   ```

4. Enter your OpenRouter API key in the environment before launching the backend:

   ```bash
   export OPENROUTER_API_KEY="sk-..."
   ```

The interface allows you to upload a CSV file, inspect planner decisions, and trigger
the `/actions` endpoints exposed by the backend service.
