"""Explicit experiment tracking, validation selection and one held-out test evaluation."""

import importlib.metadata
import logging
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

from kedro_lasso.domain import fingerprint, make_model, regression_metrics

logger = logging.getLogger(__name__)


def split_training(frame: pd.DataFrame, config: dict) -> dict:
    test_size = float(config["test_size"])
    validation_size = float(config["validation_size"])
    if not 0 < test_size < 1 or not 0 < validation_size < 1 - test_size:
        raise ValueError("test_size + validation_size must be below 1; both must be positive.")
    development, test = train_test_split(
        frame, test_size=test_size, random_state=config["random_state"]
    )
    train, validation = train_test_split(
        development,
        test_size=validation_size / (1 - test_size),
        random_state=config["random_state"],
    )
    if min(map(len, (train, validation, test))) < 2:
        raise ValueError("Each split needs at least two rows.")
    return {"train": train, "validation": validation, "test": test}


def train_experiments(
    splits: dict, schema: dict, model_config: dict, tracking: dict, split_config: dict
) -> tuple[dict, dict]:
    alphas = [float(value) for value in model_config["alphas"]]
    if not alphas or any(not np.isfinite(value) or value <= 0 for value in alphas):
        raise ValueError("model.alphas must contain finite positive values.")
    if mlflow.active_run() is not None:
        raise ValueError("Close the active MLflow run before starting a Kedro experiment sweep.")
    mlflow.set_tracking_uri(tracking["uri"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(tracking["experiment_name"])
    if experiment is None:
        # HTTP tracking uses the server's artifact proxy; local SQLite uses a file URI.
        local_store = tracking["uri"].startswith("sqlite:")
        experiment_id = client.create_experiment(
            tracking["experiment_name"],
            artifact_location=tracking["artifact_root"] if local_store else None,
        )
    else:
        experiment_id = experiment.experiment_id

    features, target = schema["features"], schema["target"]
    train, validation, test = (splits[key] for key in ("train", "validation", "test"))
    started_at = datetime.now(UTC).isoformat()
    with mlflow.start_run(experiment_id=experiment_id, run_name=tracking["run_name"]) as parent:
        run_id = parent.info.run_id
        mlflow.set_tags({"pipeline": "training", "dataset": tracking["dataset_name"]})
        mlflow.log_params(
            {
                "features": ",".join(features),
                "target": target,
                "random_state": split_config["random_state"],
                "test_size": split_config["test_size"],
                "validation_size": split_config["validation_size"],
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
            }
        )
        provenance = {name: fingerprint(data) for name, data in splits.items()}
        mlflow.log_dict(provenance, "data_fingerprints.json")
        mlflow.log_dict(
            {"data": schema, "model": model_config, "split": split_config}, "configuration.json"
        )
        mlflow.log_dict(
            {name: data[schema["id_column"]].tolist() for name, data in splits.items()},
            "split_ids.json",
        )
        trials = []
        for alpha in alphas:
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=f"alpha={alpha:g}", nested=True
            ) as child:
                candidate = make_model(alpha, model_config)
                candidate.fit(train[features], train[target])
                metrics = regression_metrics(
                    validation[target], candidate.predict(validation[features])
                )
                mlflow.log_params({"alpha": alpha, **model_config, "alphas": str(alphas)})
                mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
                mlflow.log_metric(
                    "n_nonzero_coefficients", int(np.count_nonzero(candidate[-1].coef_))
                )
                trials.append({"run_id": child.info.run_id, "alpha": alpha, **metrics})

        winner = min(trials, key=lambda trial: trial["rmse"])
        development = pd.concat([train, validation])
        model = make_model(winner["alpha"], model_config)
        model.fit(development[features], development[target])
        test_metrics = regression_metrics(test[target], model.predict(test[features]))
        baseline = DummyRegressor().fit(development[features], development[target])
        baseline_metrics = regression_metrics(test[target], baseline.predict(test[features]))
        mlflow.log_params({"best_alpha": winner["alpha"], "best_trial_run_id": winner["run_id"]})
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metrics(
            {f"baseline_test_{key}": value for key, value in baseline_metrics.items()}
        )
        mlflow.log_metric("best_validation_rmse", winner["rmse"])
        mlflow.log_dict(
            dict(zip(features, model[-1].coef_.tolist(), strict=True)), "scaled_coefficients.json"
        )
        example = development[features].head(5)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="lasso",
            serialization_format="cloudpickle",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            pip_requirements=[
                f"{package}=={importlib.metadata.version(package)}"
                for package in ("scikit-learn", "numpy", "pandas", "scipy")
            ],
        )
        report = {
            "run_id": run_id,
            "model_uri": model_info.model_uri,
            "created_at": started_at,
            "best_alpha": winner["alpha"],
            "best_validation_rmse": winner["rmse"],
            "test_metrics": test_metrics,
            "baseline_test_metrics": baseline_metrics,
            "trials": trials,
            "schema": schema,
            "fingerprints": provenance,
        }
        mlflow.log_dict(report, "training_report.json")
    logger.info(
        "Best alpha=%s; test RMSE=%.2f PLN; run=%s", winner["alpha"], test_metrics["rmse"], run_id
    )
    return {"model": model, "report": report}, report
