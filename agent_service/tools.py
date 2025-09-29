"""Tools for automated EDA generation and PyCaret experiment orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported dataset format for {path.suffix}")


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Public wrapper that loads a dataset into a dataframe."""

    return _load_dataset(dataset_path)


def _looks_like_id(column: str, series: pd.Series) -> bool:
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return unique_ratio > 0.98 or column.lower().endswith("id")


def profile_dataset(
    dataset_path: str, target: str | None = None, output_dir: str | Path = "artifacts"
) -> dict[str, str]:
    """Generate Sweetviz and YData-Profiling reports for the provided dataset."""

    try:
        from ydata_profiling import ProfileReport
    except Exception as exc:  # pragma: no cover - heavy optional dependency
        raise RuntimeError("ydata_profiling is required to generate profile reports") from exc

    try:
        import sweetviz as sv
    except Exception as exc:  # pragma: no cover - heavy optional dependency
        raise RuntimeError("sweetviz is required to generate Sweetviz reports") from exc

    df = _load_dataset(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = output_dir / "ydata_profile.html"
    profile = ProfileReport(df, title="Dataset Profile", explorative=True)
    profile.to_file(profile_path)

    sweetviz_path = output_dir / "sweetviz_report.html"
    if target and target in df.columns:
        report = sv.analyze(df, target_feat=target)
    else:
        report = sv.analyze(df)
    report.show_html(str(sweetviz_path), open_browser=False)

    return {
        "ydata_profile": str(profile_path),
        "sweetviz_report": str(sweetviz_path),
    }


def _numeric_skew(df: pd.DataFrame) -> dict[str, float]:
    numeric_cols = df.select_dtypes(include=[np.number])
    return numeric_cols.skew(numeric_only=True).fillna(0.0).to_dict() if not numeric_cols.empty else {}


def _missingness(df: pd.DataFrame) -> dict[str, float]:
    return {column: float(df[column].isna().mean()) for column in df.columns}


def recommend_pycaret_setup(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Derive PyCaret setup keyword arguments based on dataset heuristics."""

    ignore_features: list[str] = []
    missing = _missingness(df)
    skew = _numeric_skew(df)

    for column in df.columns:
        series = df[column]
        if _looks_like_id(column, series) or missing[column] > 0.4:
            ignore_features.append(column)

    task = "unsupervised"
    metric = None
    if target and target in df.columns:
        target_series = df[target]
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
            task = "regression"
            metric = "R2"
        else:
            task = "classification"
            metric = "AUC"

    transform_cols = [col for col, value in skew.items() if abs(value) > 1.0]

    setup_kwargs = {
        "target": target,
        "normalize": True,
        "transformation": bool(transform_cols),
        "remove_multicollinearity": True,
        "multicollinearity_threshold": 0.95,
        "fix_imbalance": False,
        "ignore_features": ignore_features,
        "session_id": 42,
    }

    if task == "classification" and target:
        distribution = df[target].value_counts(normalize=True)
        if not distribution.empty and distribution.min() < 0.1:
            setup_kwargs["fix_imbalance"] = True
            setup_kwargs["fold_strategy"] = "stratifiedkfold"

    return {
        "task": task,
        "metric": metric,
        "transform_columns": transform_cols,
        "setup_kwargs": setup_kwargs,
    }


def run_pycaret_experiment(
    dataset_path: str,
    target: str,
    output_dir: str | Path = "artifacts",
    additional_setup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a lightweight PyCaret experiment using the recommended configuration."""

    df = _load_dataset(dataset_path)
    recommendation = recommend_pycaret_setup(df, target)
    setup_kwargs = {**recommendation["setup_kwargs"]}
    if additional_setup:
        setup_kwargs.update(additional_setup)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if recommendation["task"] == "classification":
        from pycaret.classification import ClassificationExperiment

        experiment = ClassificationExperiment()
    elif recommendation["task"] == "regression":
        from pycaret.regression import RegressionExperiment

        experiment = RegressionExperiment()
    else:
        raise RuntimeError("Target column required for supervised PyCaret experiments")

    experiment.setup(data=df, **setup_kwargs)
    top_models = experiment.compare_models(n_select=1, sort=recommendation["metric"] or "Accuracy")
    best_model = top_models[0] if isinstance(top_models, Iterable) else top_models
    model_path = output_dir / "best_model"
    experiment.save_model(best_model, str(model_path))

    leaderboard = experiment.pull()
    leaderboard_path = output_dir / "leaderboard.json"
    leaderboard.to_json(leaderboard_path, orient="records")

    return {
        "recommendation": recommendation,
        "model_path": str(model_path) + ".pkl",
        "leaderboard_path": str(leaderboard_path),
    }


def summarize_recommendation(recommendation: dict[str, Any]) -> str:
    """Render a human readable explanation of the recommended setup."""

    setup_kwargs = recommendation["setup_kwargs"]
    summary = {
        "task": recommendation["task"],
        "metric": recommendation["metric"],
        "transformation": setup_kwargs.get("transformation"),
        "ignore_features": setup_kwargs.get("ignore_features"),
        "fix_imbalance": setup_kwargs.get("fix_imbalance"),
    }
    return json.dumps(summary, indent=2)


__all__ = [
    "load_dataset",
    "profile_dataset",
    "recommend_pycaret_setup",
    "run_pycaret_experiment",
    "summarize_recommendation",
]
