from kedro.pipeline import Pipeline

from kedro_lasso.pipelines import batch_scoring, data_engineering, data_science


def register_pipelines() -> dict[str, Pipeline]:
    preparation = data_engineering.create_pipeline()
    training = preparation + data_science.create_pipeline()
    scoring = batch_scoring.create_pipeline()
    return {
        "__default__": training + scoring,
        "prepare": preparation,
        "training": training,
        "scoring": scoring,
    }
