from kedro.pipeline import Pipeline, node, pipeline

from kedro_lasso.training import split_training, train_experiments


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                split_training,
                ["validated_training", "params:split"],
                "data_splits",
                name="split_training",
                tags=["training"],
            ),
            node(
                train_experiments,
                ["data_splits", "params:data", "params:model", "params:tracking", "params:split"],
                ["model_bundle", "training_report"],
                name="train_lasso_sweep",
                tags=["training"],
            ),
        ]
    )
