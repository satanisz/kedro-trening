"""Execute both teaching notebooks against isolated databases without altering demo data."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from kedro_lasso.commands import prepare_runtime
from kedro_lasso.settings import PROJECT_ROOT


def main():
    prepare_runtime()
    # A new directory per verification keeps all MLflow and warehouse writes isolated.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="notebook-", dir=PROJECT_ROOT / ".runtime" / "tmp") as temp:
        os.environ["KEDRO_DEMO_DATA_DIR"] = Path(temp).as_posix()
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        prepare_runtime()
        # Override argv rather than depend on a globally registered notebook kernel.
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpec

        for name in ("01_kedro_mlflow_lab.ipynb", "02_kedro_mlflow_lab_en.ipynb"):
            notebook = nbformat.read(PROJECT_ROOT / "notebooks" / name, as_version=4)
            manager = KernelManager(kernel_name="python3")
            manager._kernel_spec = KernelSpec(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="Project Python",
                language="python",
            )
            client = NotebookClient(
                notebook,
                timeout=240,
                km=manager,
                resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}},
            )
            target = PROJECT_ROOT / ".runtime" / f"executed_{name}"
            try:
                client.execute(cleanup_kc=True)
            finally:
                nbformat.write(notebook, target)
            print(f"Notebook executed successfully: {target}")


if __name__ == "__main__":
    main()
