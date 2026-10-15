"""Data contracts and model operations without database or notebook dependencies."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def validate_frame(frame: pd.DataFrame, schema: dict, *, training: bool) -> pd.DataFrame:
    features = schema["features"]
    if not features or len(set(features)) != len(features):
        raise ValueError("Feature list must be non-empty and unique.")
    if schema["target"] in features or schema["id_column"] in features:
        raise ValueError("Target and ID cannot be features (data leakage).")
    required = [schema["id_column"], *features]
    if training:
        required.append(schema["target"])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input dataset is empty.")
    result = frame.loc[:, required].copy()
    ids = result[schema["id_column"]]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("IDs must be unique and non-null.")
    if not pd.api.types.is_integer_dtype(ids):
        raise ValueError("Property IDs must be integers.")
    numeric = [*features, *([schema["target"]] if training else [])]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if np.isinf(result[column]).any() or result[column].isna().all():
            raise ValueError(f"Column {column} has infinite values or is entirely missing.")
    if training and result[schema["target"]].isna().any():
        raise ValueError("Training target cannot be missing.")
    return result


def fingerprint(frame: pd.DataFrame) -> str:
    """Capture values, columns, dtypes and order for run provenance."""
    digest = hashlib.sha256()
    digest.update(str(list(zip(frame.columns, frame.dtypes.astype(str), strict=True))).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def make_model(alpha: float, config: dict) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lasso",
                Lasso(alpha=alpha, max_iter=int(config["max_iter"]), tol=config["tolerance"]),
            ),
        ]
    )


def regression_metrics(actual, predicted) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
