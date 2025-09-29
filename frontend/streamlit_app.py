"""Streamlit interface for the PyCaret copilot backend."""

from __future__ import annotations

import uuid
from pathlib import Path

import requests
import streamlit as st

API_URL = st.secrets.get("api_url", "http://localhost:8000")


@st.cache_data
def save_uploaded_file(upload) -> Path:
    output_dir = Path("uploaded")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"{uuid.uuid4()}_{upload.name}"
    with path.open("wb") as handle:
        handle.write(upload.getbuffer())
    return path


def render_chat():
    st.title("PyCaret Copilot")
    st.write("Interact with the backend agent and trigger automated workflows.")

    uploaded_file = st.file_uploader("Dataset (CSV/Parquet)")
    dataset_path: Path | None = None
    if uploaded_file:
        dataset_path = save_uploaded_file(uploaded_file)
        st.success(f"Saved dataset to {dataset_path}")

    target = st.text_input("Target column (optional)")

    user_prompt = st.text_area("Message", "Generate an EDA report for this dataset")
    if st.button("Send"):
        messages = [{"role": "user", "content": user_prompt}]
        payload = {
            "messages": messages,
            "conversation_id": str(uuid.uuid4()),
            "dataset_path": str(dataset_path) if dataset_path else None,
            "target": target or None,
        }
        response = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
        if response.status_code != 200:
            st.error(f"Backend error: {response.text}")
            return
        data = response.json()
        st.subheader("Agent Response")
        st.write(data["reply"])
        st.write(f"Planner action: `{data['planner_action']}`")
        if data.get("artifacts"):
            st.write("Artifacts:")
            st.json(data["artifacts"])

    st.markdown("---")
    st.subheader("Manual actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate EDA", key="eda"):
            if not dataset_path:
                st.warning("Upload a dataset first")
            else:
                payload = {
                    "action": "generate_eda",
                    "dataset_path": str(dataset_path),
                    "target": target or None,
                }
                response = requests.post(f"{API_URL}/actions", json=payload, timeout=120)
                st.write(response.json())
    with col2:
        if st.button("Run PyCaret", key="pycaret"):
            if not dataset_path or not target:
                st.warning("Upload a dataset and set a target")
            else:
                payload = {
                    "action": "run_pycaret",
                    "dataset_path": str(dataset_path),
                    "target": target,
                }
                response = requests.post(f"{API_URL}/actions", json=payload, timeout=600)
                st.write(response.json())


if __name__ == "__main__":
    render_chat()
