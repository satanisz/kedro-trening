"""Execute both workshop notebooks in disposable databases using the project kernel."""

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
from nbclient import NotebookClient

from kedro_lasso.commands import prepare_runtime
from kedro_lasso.settings import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-data", action="store_true", help="Keep databases for UI inspection")
    args = parser.parse_args()
    prepare_runtime()
    targets = [
        ("PL", "01_cykl_zycia_modelu.ipynb"),
        ("EN", "01_model_lifecycle.ipynb"),
    ]
    previous = os.environ.get("KEDRO_WORKSHOP_ROOT")
    try:
        with TemporaryDirectory(
            prefix="workshops-",
            dir=PROJECT_ROOT / ".runtime" / "tmp",
            delete=not args.keep_data,
        ) as temp:
            if args.keep_data:
                print(f"Retained verification data: {temp}", flush=True)
            os.environ["KEDRO_WORKSHOP_ROOT"] = temp
            for language, name in targets:
                source = PROJECT_ROOT / "trening" / language / name
                notebook = nbformat.read(source, as_version=4)
                nbformat.validate(notebook)
                manager = KernelManager(kernel_name="python3")
                manager._kernel_spec = KernelSpec(
                    argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    display_name="Project Python",
                    language="python",
                )
                client = NotebookClient(
                    notebook,
                    timeout=300,
                    km=manager,
                    resources={"metadata": {"path": str(source.parent)}},
                )
                output = PROJECT_ROOT / ".runtime" / f"executed_workshop_{language.lower()}.ipynb"
                try:
                    client.execute(cleanup_kc=True)
                finally:
                    nbformat.write(notebook, output)
                summaries = list((Path(temp) / language.lower()).glob("*/summary.json"))
                summary = json.loads(summaries[0].read_text(encoding="utf-8"))
                assert summary["rollback_verified"]
                assert summary["retraining_improved_rmse"]
                print(
                    f"PASS {language}: training, registry, scoring, monitoring, "
                    "retraining, rollback"
                )
                print(f"Executed notebook: {output}")
    finally:
        if previous is None:
            os.environ.pop("KEDRO_WORKSHOP_ROOT", None)
        else:
            os.environ["KEDRO_WORKSHOP_ROOT"] = previous


if __name__ == "__main__":
    main()
