# Kedro + JupyterLab + MLflow + SQLite

A runnable machine-learning workflow: source SQLite database → Kedro pipeline → Lasso experiments in MLflow → predictions in a separate SQLite database. The notebook uses the same catalog, configuration, and source code as batch execution. Kedro-Viz is optional.

## Requirements

Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). The container option requires a running Docker Engine and Docker Compose.

## Installation

Run this in the project directory:

```powershell
uv sync --locked --python 3.12
```

`uv.lock` records the resolved dependencies. The stack uses Kedro 1.7.0, Kedro-Datasets 9.6.0, Kedro-Viz 12.5.0, JupyterLab 4.6.4, and MLflow 3.17.1.

## Configuration

The default configuration works without changes or external data. `conf/base/parameters.yml` defines the features, split, and `alpha` grid. `conf/base/catalog.yml` defines SQL inputs and outputs; `conf/base/globals.yml` defines paths and URIs.

Optional environment variables are `KEDRO_DEMO_DATA_DIR` for an absolute data directory, `MLFLOW_TRACKING_URI` for an external tracking server, and `KEDRO_DEMO_JUPYTER_TOKEN` for the local JupyterLab token. Put local overrides in `conf/local/`.

## Quick start

```powershell
uv run demo stack
```

The command creates a source database when needed, runs the pipeline, and starts MLflow and JupyterLab. It runs in the current terminal; `Ctrl+C` stops the services. Service logs are written to `.runtime/logs/`.

| Tool | Address |
| --- | --- |
| JupyterLab | [Open the lab](http://127.0.0.1:8888/lab?token=kedro-demo) |
| MLflow | [Open experiments](http://127.0.0.1:5000) |
| Kedro-Viz, when enabled | [Open the graph](http://127.0.0.1:4141) |

In JupyterLab, open the [Polish notebook](notebooks/01_kedro_mlflow_lab.ipynb) or the [English notebook](notebooks/02_kedro_mlflow_lab_en.ipynb), then choose **Run → Run All Cells**. In MLflow, select **Model training**, open the **housing-lasso** experiment, go to **Runs**, and expand the parent run to compare `alpha` values. The JupyterLab link uses the demo token; after changing it, use the URL printed by the launcher.

Start the optional graph as well:

```powershell
uv run demo stack --with-viz
```

When a model already exists and only the tools are needed:

```powershell
uv run demo stack --skip-run --with-viz
```

## Batch runs

```powershell
uv run demo seed
uv run kedro run
uv run demo inspect
```

The source contains 1,500 synthetic properties with a price and 250 without one. The pipeline compares six `alpha` values, selects a model on validation data, and evaluates it on a separate test set. MLflow stores parameters, metrics, configuration, data fingerprints, and the model with its imputer and scaler. The output SQLite database stores model, trial, and prediction-batch history.

| Task | Command |
| --- | --- |
| Prepare training data only | `uv run kedro run --pipeline prepare` |
| Train and save a model only | `uv run kedro run --pipeline training` |
| Score with an existing model | `uv run kedro run --pipeline scoring` |
| Use the short experiment grid | `uv run kedro run --env quick --pipeline training` |
| Change the split seed | `uv run kedro run --params split.random_state=123` |

Run `uv run demo mlflow`, `uv run demo lab`, and `uv run demo viz` in separate terminals to start individual services.

## Docker Compose

```powershell
docker compose up --build
```

Compose starts seed, the MLflow server, the pipeline, and JupyterLab in that order. The `viz` profile adds visualisation:

```powershell
docker compose --profile viz up --build
```

Run another batch job and stop the services:

```powershell
docker compose run --rm pipeline kedro run --env quick
docker compose down
```

Container data is stored in the `kedro-lasso_demo-data` volume; local execution uses `data/`. These are separate environments. `down` keeps the volume. Source code, configuration, and notebooks are mounted from the project, so changes are available without rebuilding the image. Dependency changes require a rebuild.

## Documentation and tests

[Architecture and database schema](docs/architecture.md) explain component boundaries, transactions, configuration changes, and demo limitations. The [walkthrough](docs/walkthrough.md) guides you through the notebook and MLflow.

```powershell
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python scripts/verify_notebook.py
```

Integration tests and the notebook verifier use separate temporary databases. The verifier saves both executed notebooks as `.runtime/executed_<notebook-name>.ipynb`. The CI workflow runs the same checks on Windows and Linux.

This is a local architecture demonstration: SQLite assumes few concurrent writers, and services bind to the loopback interface. See the architecture document for the path to shared infrastructure.
