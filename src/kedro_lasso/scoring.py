"""Batch inference uses the saved training schema and its fitted preprocessing."""

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from kedro_lasso.domain import fingerprint, validate_frame


def score_batch(model_bundle: dict, scoring: pd.DataFrame) -> dict:
    report = model_bundle["report"]
    schema = report["schema"]
    clean = validate_frame(scoring, schema, training=False)
    predictions = model_bundle["model"].predict(clean[schema["features"]])
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")
    return {
        "batch_id": uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "input_fingerprint": fingerprint(clean),
        "report": report,
        "predictions": pd.DataFrame(
            {"property_id": clean[schema["id_column"]].to_numpy(), "prediction": predictions}
        ),
    }
