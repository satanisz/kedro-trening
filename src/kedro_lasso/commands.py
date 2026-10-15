"""Small cross-platform launcher. Kedro remains the pipeline execution engine."""

import argparse
import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from kedro_lasso.config import ensure_data_dirs, load_config
from kedro_lasso.seed import seed_source
from kedro_lasso.settings import PROJECT_ROOT


def prepare_runtime() -> None:
    runtime = PROJECT_ROOT / ".runtime"
    for directory in ("tmp", "jupyter", "ipython", "config", "data", "matplotlib"):
        (runtime / directory).mkdir(parents=True, exist_ok=True)
    # Keep kernels, temporary files and user-level Jupyter setup inside the project.
    defaults = {
        "JUPYTER_RUNTIME_DIR": str(runtime / "jupyter"),
        "JUPYTER_CONFIG_DIR": str(runtime / "config"),
        "JUPYTER_DATA_DIR": str(runtime / "data"),
        "IPYTHONDIR": str(runtime / "ipython"),
        "MPLCONFIGDIR": str(runtime / "matplotlib"),
        "TEMP": str(runtime / "tmp"),
        "TMP": str(runtime / "tmp"),
        "KEDRO_DISABLE_TELEMETRY": "1",
        "MLFLOW_ENABLE_TELEMETRY": "false",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "DO_NOT_TRACK": "1",
    }
    for key, value in defaults.items():
        # Temp paths need an explicit override on sandboxed Windows hosts.
        if key in ("TEMP", "TMP"):
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    ensure_data_dirs()


def seed() -> None:
    uri = load_config()["credentials"]["source_db"]["con"]
    created = seed_source(uri)
    print("Source created: 1500 training rows + 250 scoring rows." if created else "Source exists.")


def run_pipeline() -> None:
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT) as session:
        session.run()


def service_command(service: str, host: str) -> list[str]:
    python = sys.executable
    globals_config = load_config()["globals"]
    if service == "mlflow":
        return [
            python,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            f"sqlite:///{globals_config['data_dir']}/mlflow/mlflow.db",
            "--artifacts-destination",
            f"{globals_config['data_dir']}/mlflow/artifacts",
            "--host",
            host,
            "--port",
            "5000",
            "--workers",
            "1",
            "--allowed-hosts",
            "localhost:*,127.0.0.1:*,mlflow:*,mlflow",
        ]
    if service == "lab":
        os.environ.setdefault(
            "JUPYTER_TOKEN", os.environ.get("KEDRO_DEMO_JUPYTER_TOKEN", "kedro-demo")
        )
        return [
            python,
            "-m",
            "kedro",
            "jupyter",
            "lab",
            "--ip",
            host,
            "--port",
            "8888",
            "--no-browser",
            "--allow-root",
            "--ServerApp.port_retries=0",
            "--ServerApp.root_dir=" + str(PROJECT_ROOT),
        ]
    if service == "viz":
        return [
            python,
            "-m",
            "kedro",
            "viz",
            "run",
            "--host",
            host,
            "--port",
            "4141",
            "--no-browser",
        ]
    raise ValueError(service)


def inspect_warehouse() -> None:
    engine = create_engine(load_config()["credentials"]["warehouse_db"]["con"])
    try:
        with engine.connect() as connection:
            if "model_runs" not in inspect(connection).get_table_names():
                print("Warehouse is empty. Run: demo seed, then kedro run.")
                return
            print(
                pd.read_sql(
                    text(
                        "SELECT run_id, best_alpha, test_rmse, test_mae, test_r2 "
                        "FROM model_runs ORDER BY created_at DESC LIMIT 5"
                    ),
                    connection,
                ).to_string(index=False)
            )
            print(
                pd.read_sql(
                    text(
                        "SELECT batch_id, run_id, n_rows, created_at "
                        "FROM prediction_batches ORDER BY created_at DESC LIMIT 5"
                    ),
                    connection,
                ).to_string(index=False)
            )
    finally:
        engine.dispose()


def stack(with_viz: bool, skip_run: bool) -> None:
    services = {"mlflow": 5000, "lab": 8888}
    if with_viz:
        services["viz"] = 4141
    # Check before training, so occupied ports do not leave an unexplained partial launch.
    for name, port in services.items():
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise SystemExit(
                    f"Port {port} ({name}) is occupied. Stop that service first."
                ) from error
    seed()
    if not skip_run:
        run_pipeline()
    logs = PROJECT_ROOT / ".runtime" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    with ExitStack() as files:
        try:
            for name in services:
                output = files.enter_context((logs / f"{name}.log").open("w", encoding="utf-8"))
                kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                children.append(
                    subprocess.Popen(
                        service_command(name, "127.0.0.1"),
                        cwd=PROJECT_ROOT,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        **kwargs,
                    )
                )
            print("MLflow:    http://127.0.0.1:5000", flush=True)
            print(
                "Jupyter:   http://127.0.0.1:8888/lab?token=" + os.environ["JUPYTER_TOKEN"],
                flush=True,
            )
            if with_viz:
                print("Kedro-Viz: http://127.0.0.1:4141", flush=True)
            print(f"Starting services; logs: {logs}. Ctrl+C stops the stack.", flush=True)
            while all(child.poll() is None for child in children):
                time.sleep(1)
            raise SystemExit(f"A service exited. Inspect logs in {logs}.")
        except KeyboardInterrupt:
            print("Stopping local stack.")
        finally:
            for child in children:
                if child.poll() is None:
                    if os.name == "nt":
                        # MLflow and Jupyter can spawn workers; clean up this process tree only.
                        subprocess.run(
                            ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        child.terminate()
            for child in children:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seed", help="Create source tables if absent; preserve existing data")
    subparsers.add_parser("inspect", help="Show latest model runs and prediction batches")
    for name in ("mlflow", "lab", "viz"):
        service = subparsers.add_parser(name, help=f"Run {name} in the foreground")
        service.add_argument("--host", default="127.0.0.1")
    supervisor = subparsers.add_parser("stack", help="Seed, train and launch the local services")
    supervisor.add_argument("--with-viz", action="store_true")
    supervisor.add_argument(
        "--skip-run", action="store_true", help="Keep existing training results"
    )
    args = parser.parse_args()
    prepare_runtime()
    if args.command == "seed":
        seed()
    elif args.command == "inspect":
        inspect_warehouse()
    elif args.command == "stack":
        stack(args.with_viz, args.skip_run)
    else:
        try:
            raise SystemExit(
                subprocess.call(service_command(args.command, args.host), cwd=PROJECT_ROOT)
            )
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
