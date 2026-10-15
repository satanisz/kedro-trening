import numpy as np
import pandas as pd
import pytest

from kedro_lasso.domain import make_model, validate_frame
from kedro_lasso.training import split_training

SCHEMA = {"id_column": "property_id", "target": "price_pln", "features": ["area_m2"]}


def frame():
    return pd.DataFrame(
        {
            "property_id": range(100),
            "area_m2": np.arange(100) + 20.0,
            "price_pln": np.arange(100) * 10000.0 + 200000.0,
        }
    )


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame(),
        frame().drop(columns="area_m2"),
        frame().assign(property_id=1),
        frame().assign(price_pln=np.nan),
        frame().assign(area_m2=np.inf),
        frame().assign(area_m2=np.nan),
    ],
)
def test_bad_input_rejected(bad):
    with pytest.raises(ValueError):
        validate_frame(bad, SCHEMA, training=True)


def test_target_cannot_be_feature():
    with pytest.raises(ValueError, match="leakage"):
        validate_frame(frame(), {**SCHEMA, "features": ["price_pln"]}, training=True)


def test_missing_features_allowed_but_missing_targets_rejected():
    data = frame()
    data.loc[0, "area_m2"] = np.nan
    assert validate_frame(data, SCHEMA, training=True).area_m2.isna().sum() == 1
    data.loc[0, "price_pln"] = np.nan
    with pytest.raises(ValueError, match="target"):
        validate_frame(data, SCHEMA, training=True)


def test_splits_disjoint_reproducible_and_exhaustive():
    config = {"test_size": 0.2, "validation_size": 0.2, "random_state": 42}
    splits = split_training(frame(), config)
    ids = [set(part.property_id) for part in splits.values()]
    assert len(ids[0]) == 60 and len(ids[1]) == len(ids[2]) == 20
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(frame().property_id)
    assert splits["test"].equals(split_training(frame(), config)["test"])


@pytest.mark.parametrize("test,validation", [(0.6, 0.5), (0.0, 0.2), (0.2, -0.1)])
def test_invalid_split_sizes_rejected(test, validation):
    with pytest.raises(ValueError):
        split_training(
            frame(), {"test_size": test, "validation_size": validation, "random_state": 42}
        )


def test_preprocessing_fitted_only_on_training():
    model = make_model(1.0, {"max_iter": 1000, "tolerance": 0.0001})
    train = pd.DataFrame({"area_m2": [1.0, 2.0, np.nan, 3.0]})
    model.fit(train, [2.0, 4.0, 4.0, 6.0])
    median = model["imputer"].statistics_.copy()
    model.predict(pd.DataFrame({"area_m2": [100000.0, np.nan]}))
    np.testing.assert_array_equal(model["imputer"].statistics_, median)
    assert median[0] == 2.0
