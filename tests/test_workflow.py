import copy

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DatasetError
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine, text

from kedro_lasso.commands import prepare_runtime
from kedro_lasso.config import load_config
from kedro_lasso.datasets import WarehouseDataset
from kedro_lasso.scoring import score_batch
from kedro_lasso.seed import seed_source
from kedro_lasso.settings import PROJECT_ROOT


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("workflow")
    patch = pytest.MonkeyPatch()
    patch.setenv("KEDRO_DEMO_DATA_DIR", root.as_posix())
    patch.delenv("MLFLOW_TRACKING_URI", raising=False)
    prepare_runtime()
    config = load_config()
    seed_source(config["credentials"]["source_db"]["con"], rows=300, scoring_rows=25)
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(
        project_path=PROJECT_ROOT,
        runtime_params={
            "model": {"alphas": [1.0, 1000.0]},
            "tracking": {"experiment_name": "test-workflow"},
        },
    ) as session:
        session.run()
    with KedroSession.create(project_path=PROJECT_ROOT) as session:
        catalog = session.load_context().catalog
        model = catalog.load("model_bundle")
        source = catalog.load("raw_scoring")
    yield {"config": config, "model": model, "source": source}
    patch.undo()


def test_end_to_end_lineage_model_and_quality(workflow):
    model, config = workflow["model"], workflow["config"]
    report = model["report"]
    mlflow.set_tracking_uri(config["globals"]["tracking_uri"])
    client = MlflowClient()
    parent = client.get_run(report["run_id"])
    assert parent.info.status == "FINISHED"
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    assert len(children) == 2
    assert all("test_rmse" not in child.data.metrics for child in children)
    assert report["best_alpha"] == min(report["trials"], key=lambda row: row["rmse"])["alpha"]
    assert report["test_metrics"]["rmse"] < report["baseline_test_metrics"]["rmse"] * 0.5
    restored = mlflow.sklearn.load_model(report["model_uri"])
    features = report["schema"]["features"]
    np.testing.assert_allclose(
        restored.predict(workflow["source"][features]),
        model["model"].predict(workflow["source"][features]),
    )
    engine = create_engine(config["credentials"]["warehouse_db"]["con"])
    try:
        with engine.connect() as conn:
            assert conn.scalar(text("SELECT COUNT(*) FROM predictions")) == 25
            assert conn.scalar(text("SELECT COUNT(*) FROM experiment_trials")) == 2
            assert conn.scalar(text("SELECT run_id FROM model_runs")) == report["run_id"]
    finally:
        engine.dispose()


def test_standalone_scoring_does_not_retrain(workflow):
    tracking = workflow["config"]["globals"]["tracking_uri"]
    client = MlflowClient(tracking_uri=tracking)
    experiment = client.get_experiment_by_name("test-workflow")
    before = len(client.search_runs([experiment.experiment_id]))
    with KedroSession.create(
        project_path=PROJECT_ROOT,
        runtime_params={"data": {"features": ["a_changed_training_configuration"]}},
    ) as session:
        session.run(pipeline_name="scoring")
    assert len(client.search_runs([experiment.experiment_id])) == before


def test_transaction_is_idempotent_and_rolls_back_bad_batch(workflow, tmp_path):
    credentials = {"con": f"sqlite:///{(tmp_path / 'warehouse.db').as_posix()}"}
    dataset = WarehouseDataset(credentials=credentials)
    batch = score_batch(workflow["model"], workflow["source"])
    dataset.save(batch)
    dataset.save(batch)
    invalid = copy.deepcopy(batch)
    invalid["predictions"] = pd.concat([batch["predictions"], batch["predictions"].iloc[:1]])
    with pytest.raises(DatasetError):
        dataset.save(invalid)
    engine = create_engine(credentials["con"])
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT COUNT(*) FROM predictions")) == 25
            assert connection.scalar(text("SELECT COUNT(*) FROM model_runs")) == 1
            assert connection.scalar(text("SELECT COUNT(*) FROM prediction_batches")) == 1
    finally:
        engine.dispose()


def test_seed_preserves_existing_source(workflow):
    assert not seed_source(workflow["config"]["credentials"]["source_db"]["con"])
