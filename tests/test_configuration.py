import pytest
from kedro.config import OmegaConfigLoader

from kedro_lasso.settings import CONFIG_LOADER_ARGS, PROJECT_ROOT


@pytest.mark.parametrize("environment", ["local", "quick"])
def test_runtime_overrides_preserve_sibling_parameters(environment):
    loader = OmegaConfigLoader(
        str(PROJECT_ROOT / "conf"),
        env=environment,
        runtime_params={"split": {"random_state": 123}, "model": {"tolerance": 0.001}},
        **CONFIG_LOADER_ARGS,
    )
    parameters = loader["parameters"]
    assert parameters["split"] == {
        "random_state": 123,
        "test_size": 0.2,
        "validation_size": 0.2,
    }
    assert parameters["model"]["tolerance"] == 0.001
    assert parameters["model"]["max_iter"] == 20000
    if environment == "quick":
        assert parameters["model"]["alphas"] == [1.0, 1000.0]
