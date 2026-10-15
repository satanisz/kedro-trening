from kedro.pipeline import Pipeline, node, pipeline

from kedro_lasso.domain import validate_frame
from kedro_lasso.scoring import score_batch


def validate_scoring(frame, model_bundle):
    return validate_frame(frame, model_bundle["report"]["schema"], training=False)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                validate_scoring,
                ["raw_scoring", "model_bundle"],
                "validated_scoring",
                name="validate_scoring",
                tags=["scoring"],
            ),
            node(
                score_batch,
                ["model_bundle", "validated_scoring"],
                "warehouse",
                name="score_and_publish_batch",
                tags=["scoring"],
            ),
        ]
    )
