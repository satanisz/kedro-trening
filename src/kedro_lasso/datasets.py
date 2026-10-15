"""A transactional output adapter: a whole scoring batch commits or rolls back."""

import json

from kedro.io import AbstractDataset
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    insert,
    select,
)


class WarehouseDataset(AbstractDataset[dict, dict]):
    """Append batches and keep model lineage; repeat saves of one batch are idempotent.

    Explicit tables avoid pandas creating unconstrained production output tables.
    The adapter is the only pipeline component that knows the warehouse connection.
    """

    def __init__(self, credentials: dict, metadata: dict | None = None):
        self._con = credentials["con"]
        self.metadata = metadata
        self._schema = MetaData()
        self._models = Table(
            "model_runs",
            self._schema,
            Column("run_id", String(64), primary_key=True),
            Column("model_uri", Text, nullable=False),
            Column("created_at", String(64), nullable=False),
            Column("best_alpha", Float, nullable=False),
            Column("test_rmse", Float, nullable=False),
            Column("test_mae", Float, nullable=False),
            Column("test_r2", Float, nullable=False),
            Column("report_json", Text, nullable=False),
        )
        self._trials = Table(
            "experiment_trials",
            self._schema,
            Column("trial_run_id", String(64), primary_key=True),
            Column("run_id", ForeignKey("model_runs.run_id"), nullable=False, index=True),
            Column("alpha", Float, nullable=False),
            Column("validation_rmse", Float, nullable=False),
        )
        self._batches = Table(
            "prediction_batches",
            self._schema,
            Column("batch_id", String(64), primary_key=True),
            Column("run_id", ForeignKey("model_runs.run_id"), nullable=False, index=True),
            Column("created_at", String(64), nullable=False),
            Column("n_rows", Integer, nullable=False),
            Column("input_fingerprint", String(64), nullable=False),
        )
        self._predictions = Table(
            "predictions",
            self._schema,
            Column("batch_id", ForeignKey("prediction_batches.batch_id"), primary_key=True),
            Column("property_id", Integer, primary_key=True),
            Column("prediction", Float, nullable=False),
        )

    def _engine(self):
        engine = create_engine(self._con)
        if engine.dialect.name == "sqlite":

            @event.listens_for(engine, "connect")
            def configure_sqlite(dbapi_connection, _):
                dbapi_connection.execute("PRAGMA foreign_keys=ON")
                dbapi_connection.execute("PRAGMA busy_timeout=30000")
                # Real transaction control is needed for transactional DDL as well as DML.
                dbapi_connection.isolation_level = None

            @event.listens_for(engine, "begin")
            def begin_sqlite(connection):
                connection.exec_driver_sql("BEGIN IMMEDIATE")

        return engine

    def _save(self, data: dict) -> None:
        engine = self._engine()
        report, batch_id = data["report"], data["batch_id"]
        run_id = report["run_id"]
        try:
            with engine.begin() as connection:
                self._schema.create_all(connection)
                if (
                    connection.scalar(
                        select(self._models.c.run_id).where(self._models.c.run_id == run_id)
                    )
                    is None
                ):
                    connection.execute(
                        insert(self._models),
                        {
                            "run_id": run_id,
                            "model_uri": report["model_uri"],
                            "created_at": report["created_at"],
                            "best_alpha": report["best_alpha"],
                            **{
                                f"test_{key}": value
                                for key, value in report["test_metrics"].items()
                            },
                            "report_json": json.dumps(report),
                        },
                    )
                    connection.execute(
                        insert(self._trials),
                        [
                            {
                                "trial_run_id": trial["run_id"],
                                "run_id": run_id,
                                "alpha": trial["alpha"],
                                "validation_rmse": trial["rmse"],
                            }
                            for trial in report["trials"]
                        ],
                    )
                connection.execute(
                    delete(self._predictions).where(self._predictions.c.batch_id == batch_id)
                )
                connection.execute(
                    delete(self._batches).where(self._batches.c.batch_id == batch_id)
                )
                connection.execute(
                    insert(self._batches),
                    {
                        "batch_id": batch_id,
                        "run_id": run_id,
                        "created_at": data["created_at"],
                        "n_rows": len(data["predictions"]),
                        "input_fingerprint": data["input_fingerprint"],
                    },
                )
                connection.execute(
                    insert(self._predictions),
                    [
                        {"batch_id": batch_id, **row}
                        for row in data["predictions"].to_dict(orient="records")
                    ],
                )
        finally:
            engine.dispose()

    def _load(self) -> dict:
        raise NotImplementedError("Use warehouse_predictions (SQLQueryDataset) to read results.")

    def _describe(self) -> dict:
        # Do not expose credentials in logs or Kedro-Viz.
        return {"tables": ["model_runs", "experiment_trials", "prediction_batches", "predictions"]}
