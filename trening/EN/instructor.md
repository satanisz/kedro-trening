# Facilitation guide and exercise solutions

The [participant guide](workshop.md) and [notebook](01_model_lifecycle.ipynb) form one
4.5-hour path. Do not assume familiarity with the interfaces. Demonstrate each click,
ask participants to repeat it, then ask them to identify the resulting evidence.

## Preparation before the workshop

1. Open a terminal in the repository root and install dependencies:

   ```powershell
   uv sync --locked --python 3.12
   uv run python -m trening.verify_workshops
   uv run pytest -q trening/test_lifecycle.py
   ```

2. Expect `PASS PL` and `PASS EN`. Executed copies are saved as
   `.runtime/executed_workshop_pl.ipynb` and `.runtime/executed_workshop_en.ipynb`.
   Their temporary databases are removed by default. Add `--keep-data` to retain
   databases for UI inspection; the verifier prints their parent directory.
3. Check availability of ports 8888 for JupyterLab and 5001 for workshop MLflow.
   Port 5000 belongs to the original demonstration.
4. Run `uv run demo lab`, open the EN notebook, and execute its first code cell.
   Copy the printed MLflow command. Select **File → New Launcher → Other → Terminal**,
   change to the repository root, and run it.
5. Do not share one `WORKSPACE` between participants. Each participant needs their
   own checkout, execution directory, and local databases.
6. Verify Python 3.12 in `.venv` and MLflow's **Model training** view.
   If `conf/local` overrides point to other databases, the notebook stops before
   writing there; use a clean project configuration.

## How to facilitate each stage

| Stage | Demonstrate | Ask | Continue when… |
| --- | --- | --- | --- |
| JupyterLab | Open a file, Shift+Enter, `[ ]` / `[*]` / `[1]`, save | What does the kernel remember? | Python and WORKSPACE paths are correct. |
| Data | `catalog.load`, missingness, contract, IDs | Is a missing feature value the same as a missing target? | Duplicate IDs and a missing column are detected. |
| Architecture | `src`, YAML, and the `training` graph | Where is training implemented? | Participants identify the node and catalog. |
| Experiments | Parent, children, validation, test, artifacts | Why is there no child test metric? | Six trials and run configuration are visible. |
| Release | Gate, Model registry, alias, manifest | Does champion automatically switch Kedro? | Both concrete version systems are identified. |
| Batch | Separate process and output database | Why does another execution have a new batch ID? | 250 records, same training run. |
| Monitoring | Two types of alerts | What can we measure without labels? | Input changes and prediction errors are distinguished. |
| Retraining | New data, fingerprints, fresh holdout | Why not compare different periods' internal tests? | Both models are evaluated on the same new records. |
| Rollback | Old manifest despite a newer bundle | Does an operational rollback always improve quality? | The batch uses the old run, then accepted v2 is restored. |

Draw **JupyterLab → KedroSession → pipeline → node → MLflow**. Dataset reads and writes
go through the catalog. JupyterLab controls the experiment; fitting stays in `src`.
MLflow integration is explicit in `training.py`; the `kedro-mlflow` plugin is not required.

The PL and EN notebooks execute equivalent code; explanatory language and workspace
subdirectory differ. Use section numbers rather than cell numbers for a mixed-language group.

## Expected results

- Initial source: 1,500 training and 250 scoring records; 900/300/300 split.
- Training: six child runs plus a parent; `alpha` selected by validation results.
- Default initial model: `alpha=100`, test RMSE approximately PLN 31,645, baseline
  approximately PLN 409,834. Do not assess exact floating-point identity across platforms.
- Fresh registry: v1 after the initial release, v2 after retraining. An MLflow 3 model
  URI can be `models:/m-...`; use the report's URI rather than manually constructing one.
- Monitoring with a PLN 120,000 price shift exceeds the PLN 50,000 quality budget.
- The challenger passes that budget and reduces RMSE by more than 50% on a fresh holdout.
- Default history: 4 batches of 250 predictions, v1 → v2 → v1 → v2.
  Additional exercise D executions increase the count and are allowed.
- `summary.json` has `rollback_verified=true` and `retraining_improved_rmse=true`;
  the final batch belongs to `active_run_id`.

## Exercise A–F solutions

### A. Missing column

Add a cell using **+**, choose **Code**, and execute:

```python
try:
    validate_frame(raw.drop(columns=["area_m2"]), schema, training=True)
except ValueError as error:
    print(error)
```

Expected: `Missing columns: ['area_m2']`. Do not write this invalid frame into the source.
The imputer handles missing values inside an existing feature column, not an absent contract column.

### B. Responsibility map

| Responsibility | File |
| --- | --- |
| Validation and sklearn Pipeline | `src/kedro_lasso/domain.py` |
| Splitting, candidates, and MLflow logging | `src/kedro_lasso/training.py` |
| Pipeline composition | `src/kedro_lasso/pipeline_registry.py` |
| Named node inputs and outputs | `src/kedro_lasso/pipelines/*/pipeline.py` |
| SQL and dataset adapters | `conf/base/catalog.yml` |
| Features, grid, and split | `conf/base/parameters.yml` |
| Transactional output | `src/kedro_lasso/datasets.py` |
| Pinned release in another process | `trening/score_release.py` |

To experiment without rewriting the model, pass `runtime_params` when creating a session,
for example `{"tracking": {"run_name": "my-sweep"}}`. Hyperparameters live in `model.alphas`;
agreed persistent changes belong in YAML. Plan grid changes before final evaluation,
not after inspecting test results.

### C. Validation-based selection

```python
winner = min(initial_report["trials"], key=lambda trial: trial["rmse"])
print(winner["alpha"], initial_report["best_alpha"])
assert winner["alpha"] == initial_report["best_alpha"]
assert all("test_rmse" not in run.data.metrics for run in children)
```

Child runs select the hyperparameter. The parent records test metrics after refitting
on train plus validation. Repeated decisions after seeing test results need a fresh final dataset.

### D. Scoring without training

After section 5, copy the printed terminal command referencing `initial.json`.
After completing the whole notebook, use the active release, `retrained.json`.

```python
before = len(client.search_runs([experiment_id]))
run_release(retrained_release)  # Use initial_release if section 7 has not run yet.
assert len(client.search_runs([experiment_id])) == before
display(warehouse_query(history_sql))
```

SQL shows a new `batch_id`, the same `run_id`, and 250 records. Rerunning the pipeline
does not update the previous batch.

### E. Drift and quality

Shifting `area_m2` triggers a distribution alert. RMSE needs actual prices. The price-to-feature
relationship can change and increase RMSE without major feature-distribution changes.
The mean/missingness heuristic does not detect every kind of drift. Real delayed labels
must be joined to retained predictions, IDs, and prediction times. Here monitoring records
are simulated locally; no external labelling system is integrated.

### F. Pinned rollback

Aliases are mutable. A manifest pins `registry_version` and `kedro_bundle_version`.
`score_release.py` checks lineage and the checksum, then supplies `load_versions` to Kedro.
The old manifest therefore uses v1 even after v2 is trained. Business quality can decline
when the previous model is unsuitable for the new regime. Rollback and promotion are
operational decisions, not a rule to choose the smallest version number.

## Final assignment solution and assessment

After completing the notebook:

```python
latest = warehouse_query(history_sql).iloc[-1]
manifest = json.loads(retrained_release.read_text(encoding="utf-8"))
assert latest["run_id"] == manifest["run_id"]
display(
    pd.Series(
        {
            "batch_id": latest["batch_id"],
            "run_id": manifest["run_id"],
            "registry_version": manifest["registry_version"],
            "kedro_bundle_version": manifest["kedro_bundle_version"],
        }
    )
)
print(f'uv run python -m trening.score_release --release "{retrained_release}"')
```

Configuration and split-ID artifacts are attached to the MLflow parent run. If a challenger
fails the gate, retain the current release, investigate, and agree on a business response.
Do not promote it merely because the old model has a monitoring alert.

Award one point each for: contracts and disjoint data; validation-based selection;
run/artifact/version identity; batch without training; correct interpretation of both
alerts; retraining and fresh holdout; rollback evidence; awareness of production gaps.
Passing is **6/8**, with batch without training and an identifiable rollback mandatory.

## Production and retirement discussion

Before real deployment, discuss time/group-aware validation, delayed-label and segment
monitoring, immutable code/dependencies, data rights, approval records, authentication,
PostgreSQL and object storage, migrations, orchestration, backups, and retention.
This workshop does not implement those production capabilities.

To retire a model, stop future batch jobs or switch consumers, record the decision's
reason and owner, retain required evidence, and delete artifacts only according to an
agreed retention policy. Here v1 remains `rollback_available` and v2 is `active`.
Do not delete v1: the exercise requires restoring the previous release.

## Shutdown

Ask participants to save notebooks and preserve manifests. Stop your own MLflow server
with Ctrl+C in its terminal, then JupyterLab with Ctrl+C in the launcher terminal.
Do not stop someone else's processes. After a fresh notebook start, point MLflow at
the newly printed database path. The participant guide contains detailed recovery steps.
