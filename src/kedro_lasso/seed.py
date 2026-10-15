"""Create reproducible synthetic housing data; never overwrite an existing source."""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect


def seed_source(uri: str, rows: int = 1500, scoring_rows: int = 250, seed: int = 42) -> bool:
    engine = create_engine(uri)
    try:
        with engine.begin() as connection:
            present = set(inspect(connection).get_table_names())
            required = {"training_properties", "scoring_properties"}
            if required <= present:
                return False
            if required & present:
                raise ValueError("Source is incomplete. Restore it or use a new source_uri.")
            rng = np.random.default_rng(seed)
            count = rows + scoring_rows
            frame = pd.DataFrame(
                {
                    "property_id": np.arange(1, count + 1),
                    "area_m2": rng.uniform(25, 180, count),
                    "rooms": rng.integers(1, 7, count).astype(float),
                    "building_age": rng.uniform(0, 90, count),
                    "distance_km": rng.uniform(0.5, 35, count),
                    "energy_score": rng.uniform(10, 100, count),
                    "noise_feature": rng.normal(0, 1, count),
                }
            )
            frame["price_pln"] = (
                150000
                + 8500 * frame.area_m2
                + 18000 * frame.rooms
                - 1800 * frame.building_age
                - 6500 * frame.distance_km
                + 900 * frame.energy_score
                + rng.normal(0, 30000, count)
            )
            # Missing values expose the fitted imputer in the sklearn pipeline.
            for column in ("building_age", "energy_score"):
                frame.loc[rng.random(count) < 0.03, column] = np.nan
            frame.iloc[:rows].to_sql(
                "training_properties", connection, index=False, if_exists="fail"
            )
            frame.iloc[rows:].drop(columns="price_pln").to_sql(
                "scoring_properties", connection, index=False, if_exists="fail"
            )
        return True
    finally:
        engine.dispose()
