"""Workshop policy, drift signals, and isolation guards, beyond the notebook happy path."""

import json

import numpy as np
import pandas as pd
import pytest

from trening.lifecycle import monitor_inputs, quality_gate, validate_workspace_connections
from trening.score_release import score_release


@pytest.mark.parametrize(
    ("rmse", "baseline", "passed"),
    [
        (30_000, 100_000, True),
        (50_000, 100_000, True),
        (50_001, 100_000, False),
        (30_000, 40_000, False),
        (np.nan, 100_000, False),
        (np.inf, 100_000, False),
        (30_000, 0, False),
        (30_000, np.nan, False),
        (-1, 100_000, False),
    ],
)
def test_release_gate_rejects_bad_and_invalid_metrics(rmse, baseline, passed):
    report = {
        "test_metrics": {"rmse": rmse},
        "baseline_test_metrics": {"rmse": baseline},
    }
    policy = {"max_rmse_pln": 50_000, "max_baseline_ratio": 0.70}
    assert quality_gate(report, policy)["passed"] is passed


def test_input_monitor_separates_mean_shift_from_missingness():
    reference = pd.DataFrame({"area": np.arange(100, dtype=float)})
    stable = monitor_inputs(reference, reference.copy(), ["area"])
    assert not bool(stable.loc["area", "alert"])
    shifted = monitor_inputs(reference, reference + 50, ["area"])
    assert bool(shifted.loc["area", "alert"])
    missing = reference.copy()
    missing.loc[:19, "area"] = np.nan
    result = monitor_inputs(reference, missing, ["area"])
    assert result.loc["area", "missing_rate_change"] == pytest.approx(0.2)
    assert bool(result.loc["area", "alert"])


def test_workspace_guard_rejects_external_connections(tmp_path):
    config = {
        "credentials": {
            "source_db": {"con": f"sqlite:///{tmp_path.as_posix()}/source/source.db"},
            "warehouse_db": {"con": f"sqlite:///{tmp_path.as_posix()}/warehouse/warehouse.db"},
        }
    }
    validate_workspace_connections(config, tmp_path)
    config["credentials"]["source_db"]["con"] = "sqlite:///another-project.db"
    with pytest.raises(ValueError, match="outside the workshop"):
        validate_workspace_connections(config, tmp_path)


def test_rejected_release_never_opens_a_model_or_writes_predictions(tmp_path):
    manifest = tmp_path / "rejected.json"
    manifest.write_text(json.dumps({"quality_gate": {"passed": False}}), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        score_release(manifest)


def test_changed_bundle_is_rejected_before_deserializing_it(tmp_path):
    model_file = tmp_path / "models" / "lasso.pkl" / "v1" / "lasso.pkl"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"not the originally approved artifact")
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps(
            {
                "quality_gate": {"passed": True},
                "data_dir": str(tmp_path),
                "kedro_bundle_version": "v1",
                "bundle_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        score_release(manifest)
