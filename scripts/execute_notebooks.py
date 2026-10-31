"""Ejecutar notebooks de entrega y guardar sus salidas verificadas."""

import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ["PATH"] = (
    str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
)
paths = [ROOT / "main.ipynb"] + [
    ROOT / "notebooks" / f"{name}.ipynb"
    for name in (
        "05_rnn_seq2seq",
        "06b_nhits_tcn_deepar_chronos",
        "07b_experimentos",
        "08_comparacion",
    )
]
executed = []
for path in paths:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(nb, path)
    executed.append(
        {
            "notebook": str(path.relative_to(ROOT)),
            "celdas_ejecutadas": sum(c.cell_type == "code" for c in nb.cells),
            "errores": 0,
        }
    )
    print(f"Ejecutado: {path.relative_to(ROOT)}", flush=True)
(ROOT / "outputs" / "runs" / "notebook_execution.json").write_text(
    json.dumps(executed, indent=2, ensure_ascii=False)
)
