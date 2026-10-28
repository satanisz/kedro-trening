"""Small workshop utilities; training and scoring stay in the Kedro project."""

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from kedro_lasso.seed import seed_source


def create_workspace(project_root: Path, language: str) -> Path:
    """Keep every notebook execution in its own directory, preserving earlier runs."""
    parent = Path(
        os.environ.get("KEDRO_WORKSHOP_ROOT", project_root / ".runtime" / "trening")
    ).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    workspace = parent / language / f"{stamp}-{uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    os.environ["KEDRO_DEMO_DATA_DIR"] = workspace.as_posix()
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{workspace.as_posix()}/mlflow/mlflow.db"
    return workspace


def quality_gate(report: dict, policy: dict) -> dict:
    """Apply the pre-agreed release policy after validation-based model selection."""
    rmse = float(report["test_metrics"]["rmse"])
    baseline = float(report["baseline_test_metrics"]["rmse"])
    checks = {
        "finite_metrics": bool(np.isfinite([rmse, baseline]).all() and rmse >= 0 and baseline > 0),
        "within_rmse_budget": bool(np.isfinite(rmse) and rmse <= policy["max_rmse_pln"]),
        "beats_baseline": bool(
            np.isfinite([rmse, baseline]).all()
            and baseline > 0
            and rmse <= baseline * policy["max_baseline_ratio"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "policy": dict(policy)}


def validate_workspace_connections(config, workspace: Path) -> None:
    """Stop before writing if local Kedro overrides point outside this exercise."""
    for key, location in (
        ("source_db", "source/source.db"),
        ("warehouse_db", "warehouse/warehouse.db"),
    ):
        expected = f"sqlite:///{workspace.as_posix()}/{location}"
        if config["credentials"][key]["con"] != expected:
            raise ValueError(
                f"{key} is overridden outside the workshop. Use a clean local configuration."
            )


def monitor_inputs(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]):
    """Illustrate mean and missingness changes; this is not a statistical drift test."""
    scale = reference[features].std(ddof=0).replace(0, np.nan)
    result = pd.DataFrame(
        {
            "mean_shift_in_reference_std": (
                (current[features].mean() - reference[features].mean()).abs() / scale
            ),
            "missing_rate_change": (
                current[features].isna().mean() - reference[features].isna().mean()
            ).abs(),
        }
    )
    result["alert"] = (
        (result["mean_shift_in_reference_std"] > 0.5)
        | (result["missing_rate_change"] > 0.10)
        | result["mean_shift_in_reference_std"].isna()
    )
    return result


def future_frame(
    workspace: Path,
    name: str,
    *,
    seed: int,
    id_offset: int,
    rows: int = 600,
    price_shift: float = 120_000,
) -> pd.DataFrame:
    """Reuse the project's generator to simulate a distinct future labelled period."""
    directory = workspace / "simulations"
    directory.mkdir(exist_ok=True)
    uri = f"sqlite:///{(directory / f'{name}.db').as_posix()}"
    seed_source(uri, rows=rows, scoring_rows=25, seed=seed)
    engine = create_engine(uri)
    try:
        with engine.connect() as connection:
            frame = pd.read_sql(text("SELECT * FROM training_properties"), connection)
    finally:
        engine.dispose()
    frame["property_id"] += id_offset
    frame["price_pln"] += price_shift
    return frame
