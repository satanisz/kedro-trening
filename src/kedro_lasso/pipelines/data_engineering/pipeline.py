from kedro.pipeline import Pipeline, node, pipeline

from kedro_lasso.domain import validate_frame


def validate_training(frame, schema):
    return validate_frame(frame, schema, training=True)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                validate_training,
                ["raw_training", "params:data"],
                "validated_training",
                name="validate_training",
                tags=["prepare", "training"],
            ),
        ]
    )
