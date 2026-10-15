"""Resolve the same Kedro configuration for operational commands and notebooks."""

from pathlib import Path

from kedro.config import OmegaConfigLoader

from kedro_lasso.settings import CONFIG_LOADER_ARGS, PROJECT_ROOT


def load_config(env: str = "local") -> OmegaConfigLoader:
    return OmegaConfigLoader(str(PROJECT_ROOT / "conf"), env=env, **CONFIG_LOADER_ARGS)


def ensure_data_dirs() -> None:
    root = Path(load_config()["globals"]["data_dir"])
    for name in ("source", "warehouse", "mlflow/artifacts", "intermediate", "models", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)
