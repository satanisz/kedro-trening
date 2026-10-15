# Demonstration walkthrough

## 1. First run

```powershell
uv sync --locked --python 3.12
uv run demo stack --with-viz
```

The launcher executes the pipeline and then starts the tools. It prints service URLs and log locations. If a service fails to start, inspect `.runtime/logs/`.

## 2. JupyterLab as a project client

Open [JupyterLab](http://127.0.0.1:8888/lab?token=kedro-demo), then open `notebooks/02_kedro_mlflow_lab_en.ipynb`. Use the project environment kernel. `demo lab` configures it by default, and the notebook explicitly loads the Kedro extension.

Run the cells in order or choose **Run → Run All Cells**. Observe:

1. SQLite loading through `catalog.load('raw_training')`.
2. Reuse of `validate_frame()` from `src/` during exploration.
3. A complete pipeline run in a new session with a different `alpha` grid.
4. Reading the report and comparing regularisation with model coefficients.
5. Reading the output database through the catalog.
6. Scoring without training and restoring a model through MLflow without Kedro.

After the training cells finish, switch to MLflow to inspect the new runs. After changing code or YAML, use `%reload_kedro` and create a new session for execution.

## 3. What to inspect in MLflow

Open [MLflow](http://127.0.0.1:5000), switch to **Model training**, select the `housing-lasso` experiment, and open **Runs**. The default GenAI view exposes tools that are not relevant to this regression workflow.

- `lasso-sweep` comes from the YAML configuration; `notebook-sweep` comes from the notebook.
- Each parent run has child runs named `alpha=...`. Compare `validation_rmse` and the number of non-zero coefficients.
- The parent run records `best_alpha`, `test_rmse`, `test_mae`, `test_r2`, and baseline metrics.
- Its artifacts contain configuration, fingerprints, split IDs, coefficients, and a report. The model is logged as `lasso`.

For `seed=42`, the default grid selects `alpha=100`, with a test RMSE of about 31.6 thousand PLN and R² of about 0.994. Compare it with the baseline and remember that the data is synthetic. A different grid or split can change selection and metrics.

## 4. Output database and independent scoring

In another terminal:

```powershell
uv run demo inspect
uv run kedro run --pipeline scoring
uv run demo inspect
```

This creates another batch of 250 predictions for the same model without creating new training runs. Query the local SQLite database with:

```sql
SELECT b.batch_id, b.run_id, b.n_rows, m.best_alpha, m.test_rmse
FROM prediction_batches b
JOIN model_runs m ON m.run_id = b.run_id
ORDER BY b.created_at DESC;

SELECT p.property_id, p.prediction, b.run_id
FROM predictions p
JOIN prediction_batches b ON b.batch_id = p.batch_id
ORDER BY b.created_at DESC, p.property_id
LIMIT 10;
```

The local database is `data/warehouse/warehouse.db`. With Compose, it is in the shared volume at `/app/data/warehouse/warehouse.db`.

## 5. Optional graph

In [Kedro-Viz](http://127.0.0.1:4141), select the default, `training`, or `scoring` pipeline. Notice that scoring depends on a saved model and source data, not on the notebook. Nodes have tags; the catalog defines the raw, intermediate, models, scoring, and reporting layers.

## 6. Make a small configuration-only change

```powershell
uv run kedro run --env quick --pipeline training
uv run kedro run --pipeline scoring
```

The `quick` environment narrows the grid to two `alpha` values. Training saves a new model version, and scoring loads the latest version. Run `uv run kedro run` to return to the full training grid.

Stop the local stack with `Ctrl+C` in its terminal. For Compose, run `docker compose down`; data remains in the volume.
