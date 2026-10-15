"""Project configuration; paths are independent of the notebook's working directory."""

import os
from pathlib import Path

from kedro.config import OmegaConfigLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "merge_strategy": {"parameters": "soft"},
    "custom_resolvers": {
        "project_root": lambda: PROJECT_ROOT.as_posix(),
        "file_uri": lambda value: Path(value).resolve().as_uri(),
        "env": lambda name, default: os.environ.get(name, default),
    },
}
