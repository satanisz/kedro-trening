"""Run the existing Kedro scoring pipeline using an explicitly approved model version."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import Version
from kedro_datasets.pickle import PickleDataset
from mlflow.tracking import MlflowClient

from kedro_lasso.commands import prepare_runtime
from kedro_lasso.config import load_config
from kedro_lasso.settings import PROJECT_ROOT
from trening.lifecycle import validate_workspace_connections


def score_release(manifest_path: Path) -> dict:
    """Verify the manifest's lineage before opening a writing Kedro session."""
    release = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not release["quality_gate"]["passed"]:
        raise ValueError("This release did not pass the quality gate.")
    root = Path(release["data_dir"])
    version = release["kedro_bundle_version"]
    bundle_file = root / "models" / "lasso.pkl" / version / "lasso.pkl"
    if hashlib.sha256(bundle_file.read_bytes()).hexdigest() != release["bundle_sha256"]:
        raise ValueError("The saved Kedro bundle does not match the approved checksum.")
    # These are trusted artifacts created locally during the workshop.
    bundle = PickleDataset(
        filepath=(root / "models" / "lasso.pkl").as_posix(), version=Version(version, None)
    ).load()
    if bundle["report"]["run_id"] != release["run_id"]:
        raise ValueError("The Kedro bundle and release refer to different training runs.")
    if bundle["report"]["model_uri"] != release["model_uri"]:
        raise ValueError("The Kedro bundle and release refer to different MLflow models.")
    client = MlflowClient(
        tracking_uri=release["tracking_uri"], registry_uri=release["tracking_uri"]
    )
    registered = client.get_model_version(release["registered_name"], release["registry_version"])
    if registered.run_id != release["run_id"]:
        raise ValueError("The registry version and release refer to different training runs.")
    os.environ["KEDRO_DEMO_DATA_DIR"] = root.as_posix()
    os.environ["MLFLOW_TRACKING_URI"] = release["tracking_uri"]
    prepare_runtime()
    validate_workspace_connections(load_config(), root)
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT) as session:
        session.run(pipeline_name="scoring", load_versions={"model_bundle": version})
    return {"run_id": release["run_id"], "kedro_bundle_version": version}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score_release(args.release), indent=2))


if __name__ == "__main__":
    main()
