# Workshop: the complete model lifecycle, click by click

**Duration:** 4 hours 30 minutes, including breaks. **Audience:** people with basic Python
knowledge who are new to Kedro, MLflow, and JupyterLab. All data is synthetic.
The notebook is a complete worked example; do exercises A–F in additional cells.
Open the [instructor guide and solutions](instructor.md) after trying the exercises yourself.

By the end, you can read datasets through Kedro, start training from a notebook, compare
experiments, release a concrete version, run a batch outside the notebook, and respond
to input changes, declining quality, and deployment incidents.

## 1. What each tool does

| Term | Meaning in this exercise |
| --- | --- |
| JupyterLab | Browser application for opening files, executing cells, and viewing results. |
| Notebook | An `.ipynb` file containing explanations and code cells executed by a kernel. |
| Kernel | The Python process holding notebook variables. Restarting clears those variables. |
| Kedro node | A Python function with named inputs and outputs. |
| Pipeline | A dependency graph between nodes, executed by Kedro. |
| Data Catalog | Dataset name → loading/saving configuration; here SQLite, Parquet, and pickle. |
| Parameters | YAML settings such as features, split proportions, and the `alpha` grid. |
| MLflow experiment | A group of related runs; here `workshop-lifecycle`. |
| MLflow run | One recorded execution with parameters, metrics, and artifacts. |
| Artifact | A file associated with a run, such as a model, configuration, or report. |
| Model Registry | Versions of logged models, separate from the run list. |
| Alias | A mutable reference to a version, such as `champion`. |
| Release manifest | A JSON file pinning versions, lineage, and the quality-gate decision. |
| Batch / scoring | Predicting prices for an entire table without training. |
| Baseline | A simple reference; here a model predicting the mean price. |
| RMSE / MAE / R² | Root mean squared error, mean absolute error, and goodness of fit; lower RMSE/MAE is better. |
| Drift / retraining / rollback | A distribution or relationship change / fitting again / returning to an earlier release. |

Lasso is linear regression with a penalty on large coefficients. Larger `alpha` means
stronger regularisation; some coefficients can become zero. Neither the smallest nor
the largest `alpha` is automatically the best choice.

## 2. Schedule

| Elapsed time | Activity | Completion evidence |
| --- | --- | --- |
| 00:00–00:20 | Installation and JupyterLab | Notebook open, correct Python. |
| 00:20–00:50 | Goal, data, and contract | 1,500/250 records and a controlled failure. |
| 00:50–01:25 | Architecture and Kedro in a notebook | Disjoint 900/300/300 splits and graph. |
| 01:25–01:35 | Break | |
| 01:35–02:10 | Experiments and MLflow | Parent plus six child runs. |
| 02:10–02:40 | Quality gate and registry | Model version, `champion`, and manifest. |
| 02:40–03:05 | Batch deployment | 250 predictions without new training. |
| 03:05–03:15 | Break | |
| 03:15–03:45 | Monitoring | Separate input and RMSE alerts. |
| 03:45–04:15 | Retraining and rollback | New model and a batch using the old version. |
| 04:15–04:30 | Final assignment and discussion | Latest batch traced to its model run. |

## 3. Prepare your computer and start JupyterLab

1. On Windows, open **File Explorer** (`Win+E`) and navigate to the project directory.
   On this computer it is `C:\Users\stani\Projects\kedro_trening`.
2. Click the address bar, enter `powershell`, and press **Enter**. The terminal must be
   in the directory containing `pyproject.toml`.
3. Paste the following command and press **Enter**. The first installation needs network
   access. If `uv` is not found, follow the
   [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
   open a new terminal, and try again.

   ```powershell
   uv sync --locked --python 3.12
   ```

4. Wait for installation to finish and the prompt to return. Then run:

   ```powershell
   uv run demo lab
   ```

5. Leave this terminal open. In your browser, open
   [JupyterLab](http://127.0.0.1:8888/lab?token=kedro-demo).
   If you configured your own token, use the URL printed by the launcher.
6. In JupyterLab's left file browser, double-click **trening**, then **EN**, then
   **01_model_lifecycle.ipynb**.
7. To read this guide in JupyterLab, right-click `workshop.md` in the same directory
   and select **Open With → Markdown Preview**. Return to the notebook tab.
8. If a kernel selection dialog appears, choose Python from the project environment.
   If uncertain, choose an available Python and check the `Python:` output after the
   first code cell. Its path must point into this project's `.venv`. Otherwise use
   **Kernel → Change Kernel…** and select the project kernel installed by `demo lab`.

On Linux/macOS, open a terminal in the repository root and run the same `uv` commands.

## 4. Learn how to execute cells

1. Explanatory cells do not execute Python. Code cells have a `[ ]` prompt on the left.
2. Click inside the first code cell in section **0** and press **Shift+Enter**.
   Alternatively select **Run → Run Selected Cell**.
3. A running cell shows `[*]`. Wait until it becomes a number such as `[1]`.
4. Results appear below the code. Do not type printed results back into the cell.
5. Copy the `WORKSPACE` path into your notes. Every fresh execution has a different path.
6. Execute the code cell in section **1**. Expect pipeline names `prepare`, `training`,
   `scoring`, `__default__`, and dataset names including `raw_training`.
7. Continue in order from top to bottom. Do not press **Shift+Enter** repeatedly while
   execution is still in progress. **Run → Run All Cells** executes the whole example,
   but during the workshop pause at each explanation and exercise.
8. To add an exercise cell, select an existing cell, click **+** above the notebook
   (tooltip **Insert a cell below**), choose **Code** in the cell-type dropdown, enter
   your code, and press **Shift+Enter**. Do not overwrite the main example cells.
9. Save using **File → Save Notebook** or **Ctrl+S**.

## 5. Data, contracts, and architecture — notebook section 2

1. Execute the `catalog.load` cell. Expect a preview, statistics, missing-value rates,
   and an area-versus-price plot. There are 1,500 training and 250 scoring records.
2. Locate `property_id`, `price_pln`, and `area_m2`. Scoring prices are unavailable.
3. Execute **A controlled failure**. `EXPECTED VALIDATION ERROR` is a successful outcome:
   the contract detected a duplicate identifier.
4. Do **exercise A**: in an additional cell remove `area_m2` and call `validate_frame`.
   Catch `ValueError` as shown in the example. Expect a message naming the missing column.
5. Execute the splitting cell. Expect 900 train, 300 validation, and 300 test records.
   Assertions verify that no ID belongs to two splits.
6. For **exercise B**, click the folder icon on the left and the root directory icon in
   the breadcrumb. Open **src → kedro_lasso**, then double-click `domain.py`, `training.py`,
   and `pipeline_registry.py`. Each opens in an editor tab.
7. Locate `SimpleImputer`, `StandardScaler`, and `Lasso` in `domain.py`, `mlflow.start_run`
   in `training.py`, and pipeline composition in `pipeline_registry.py`.
   Return to the notebook. You do not need to edit these source files.

**Pause and explain:** why could filling missing values using the entire dataset before
splitting expose information from test data?

## 6. Training and MLflow — section 3

1. Execute the cell containing `training_session.run(pipeline_name="training")` and wait
   for completion. Kedro logs and warnings about serialising your own model are expected.
2. Below it, find six `alpha` trials, Lasso and baseline metrics, the Kedro version,
   `MLflow run`, and `Model URI`. Keep the `run_id` as the identity of this trained model.
3. Return to section **0** output. Copy the **whole** command starting with
   `uv run python -m mlflow server`. It includes your execution path: do not copy a
   neighbour's command or reuse a previous notebook execution's command.
4. Select **File → New Launcher**, then the **Terminal** tile under **Other**.
   In the new terminal type `pwd` and press Enter. If necessary, change to the project
   root with `cd "C:\Users\stani\Projects\kedro_trening"`, or your own project path.
5. Paste the copied command and press **Enter**. Leave the terminal open. When the server
   reports readiness, open [the workshop MLflow UI](http://127.0.0.1:5001).
6. Click **Model training** in the top-left area selector if the GenAI view is active.
   Select **Experiments**, then click **workshop-lifecycle** in the experiment list.
7. Under **Runs**, locate **initial-sweep**. Click its left-hand arrow to expand six
   `alpha=...` child runs. Refresh the page if the new runs have not appeared.
8. Click an `alpha=...` run and inspect `alpha` and `validation_rmse` in its details.
   Go back to the list, open **initial-sweep**, and find `best_alpha`, `test_rmse`,
   `baseline_test_rmse`, and the logged `lasso` model.
9. In the parent run, click **Artifacts**. Click `configuration.json`, `split_ids.json`,
   `data_fingerprints.json`, and `training_report.json` to preview them.
10. **Exercise C:** record the winning `alpha` and identify which runs contain test
    metrics. Back in JupyterLab, run the `client.search_runs` cell and compare with the UI.

**Expected interpretation:** six child runs contain validation results; the parent
records the final evaluation. The default example selects `alpha=100` with test RMSE
around PLN 31,600. Correctness depends on the checks and agreed thresholds, not identical log text.

## 7. Gate, registry, and promotion — section 4

1. Run the `quality_gate` cell. Expect `passed: True`. A copied report with an artificial
   RMSE of 200,000 is rejected. Do not bypass a failed gate by changing a JSON flag.
2. The same cell registers the accepted model and sets `champion`. In a new database,
   this is registry version **1**. The timestamp-like Kedro version is a different version system.
3. In MLflow's left sidebar choose **Model registry**, not the experiment's **Models** tab.
   On a first visit, dismiss the introduction by clicking **Try it now**.
   Open **workshop-housing-lasso**, then **Version 1**. Inspect its source run, the
   `champion` alias, and the `kedro_bundle_version` tag. The notebook controls aliases
   during this exercise; do not edit them manually in the UI.
4. Return to the notebook and run **Record the release decision**. Expect the path to
   `releases/initial.json` and its contents.
5. In the parent run's MLflow view, open **Artifacts → releases → initial.json**.
   Compare `run_id`, `registry_version`, `kedro_bundle_version`, and the gate result.

**Checkpoint:** registration creates a version; promotion points an alias at it;
deployment executes work using a concrete artifact. These are three distinct operations.

## 8. Batch deployment — section 5

1. Run `run_release(initial_release)`. It starts a separate process using the existing
   `scoring` pipeline and the manifest's concrete bundle version.
2. Execute the warehouse query cell. Expect one batch with `n_rows=250`, its `run_id`,
   and sample predictions. An assertion compares the registry and Kedro models.
3. Check MLflow's **Runs** list. There must be no new `initial-sweep`.
4. For **exercise D**, copy the printed `Terminal command: uv run python -m trening.score_release…`.
   Open **File → New Launcher → Other → Terminal**, check `pwd`, paste the command,
   and press Enter. Wait for the prompt to return.
5. Return to the notebook, add a new cell containing `display(warehouse_query(history_sql))`,
   and execute it. There is another batch with the same `run_id`. Every batch has 250 predictions.

Saving the same prepared batch is idempotent in the SQL adapter, while rerunning the
scoring pipeline creates a **new** batch ID. These are different behaviours.

## 9. Monitoring — section 6

1. Execute `shifted_inputs`. Locate `area_m2` in the table; `alert` must be `True`.
   Only an in-memory copy changes, so the database remains usable.
2. Execute **Prices from a new period have arrived**. The old model's RMSE exceeds
   PLN 50,000. The simulation has raised prices by PLN 120,000 while retaining similar features.
3. In MLflow select **Experiments → workshop-monitoring → labelled-future-period**.
   Inspect `observed_rmse`, the `model_run_id` tag, and `input_monitor.json`.
4. For **exercise E**, explain two situations: features changed but prices are unavailable;
   features look similar but newly received prices reveal declining model quality.

Input change alone does not prove declining quality. Do not promote a model solely
because of an input alert. Monitoring without labels and with labels are different checks.

## 10. Retraining, fresh holdout, and rollback — sections 7–9

1. Execute section **7**. It preserves old source data as Parquet and replaces tables
   only inside your exercise database. The new run is `retrained-sweep`, with different fingerprints.
2. Execute **Challenger versus champion**. Evaluation uses a fresh holdout with different
   IDs and seed from training and monitoring. Comparing each model's different internal
   test results alone would not evaluate both on the new period.
3. Expect the new model's RMSE to stay under PLN 50,000 and improve by at least 50%
   versus the old model on that same holdout. Passing creates registry **Version 2**,
   `releases/retrained.json`, the new `champion`, and a batch linked to the v2 run.
4. In MLflow open **Model registry → workshop-housing-lasso** and refresh. Inspect
   versions 1 and 2. `previous` points to v1; `champion` and `candidate` point to v2.
5. Run section **8**. This controlled rollback executes the old manifest, so scoring uses
   v1 even though the newest bundle on disk is v2.
6. Compare the table's latest `run_id` with `initial_report['run_id']`.
   `Rollback verified` also confirms the actual predictions match the previous model.
7. For **exercise F**, explain why aliases and pinned manifests must be handled separately.
   In this new price regime, returning to v1 reduces quality; this is an operational drill.
8. Run section **9** to restore accepted v2 and save `summary.json`. Without extra exercise
   executions, there are **4 batches of 250 predictions**. Exercise D increases the count;
   that is expected. The final batch must reference `active_run_id` in the summary.

## 11. Final assignment — 15 minutes

Use additional cells to prepare a short note and a query/table answering:

1. Which `run_id`, registry version, and Kedro version produced the latest batch?
2. Where are the experiment configuration and split IDs stored?
3. How do you execute another batch without training? Provide the command with the correct manifest.
4. What do you do if monitored RMSE is high but the candidate fails its quality gate?
5. What must be added before team deployment with real data?

Save the notebook. Show the instructor the latest batch, both manifests, and rollback evidence.

## 12. Shut down and troubleshoot

1. Save with **Ctrl+S**. To download your copy, right-click the notebook in the left file
   browser and choose **Download**.
2. Select the MLflow terminal tab and press **Ctrl+C**. Wait for the process to stop.
3. Press **Ctrl+C** in the terminal running JupyterLab; answer `y` if Jupyter asks for
   confirmation. Preserve your execution directory as exercise evidence.
4. To restart from the beginning, select **Kernel → Restart Kernel and Clear Outputs of
   All Cells…**, confirm, and execute from section 0. A new directory is created.
   Stop the previous MLflow server and use the new execution's printed server command.

| Symptom | Recovery |
| --- | --- |
| Port 8888 page does not load | Check the `uv run demo lab` terminal, wait for startup, and use the logged URL. |
| Port 8888 or 5001 is occupied | Avoid starting a second copy. Stop your previous process in its terminal with Ctrl+C; do not terminate someone else's service. |
| `ModuleNotFoundError` | Check the `Python:` path into `.venv`, `uv sync`, and the selected kernel. |
| `NameError: catalog` / `WORKSPACE` | Execute sections 0 and 1; an earlier cell was skipped or the kernel restarted. |
| Experiment missing in MLflow | Check port 5001, completion of section 3, and the server command's database path. Port 5000 belongs to another demo. |
| Registry / metrics missing | Choose Model training and widen the window. Use Model registry for versions and the parent run for test metrics. |
| `Close the active MLflow run` | Run `mlflow.end_run()` in a new cell, then retry; do not wrap Kedro sessions in another active run. |
| `database is locked` | Wait for writes to finish and run one training/scoring job at a time. |
| Red traceback | Read its final line. Controlled failures print `EXPECTED…`; do not skip other errors. |
| UI shows the old champion | Refresh and verify that the UI uses the current `WORKSPACE` database. |
| Gate fails | Stop the release and inspect data and reports. Do not remove assertions or relax thresholds after seeing the test results. |

## Sources and extension

[Project architecture](../../docs/architecture.md),
[Kedro and Jupyter](https://docs.kedro.org/en/stable/integrations-and-plugins/notebooks_and_ipython/kedro_and_notebooks/),
[MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/workflow/).
Examples target the versions in `uv.lock`, rather than an arbitrary latest installation.
Monitoring, promotion, and batch execution are manual. Production requires real labels,
time-aware validation, access control, orchestration, migrations, backups, and retention policies.
