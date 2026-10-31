#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${LAB09_INHIBITED:-0}" != 1 ]] && command -v systemd-inhibit >/dev/null; then
    exec systemd-inhibit --what=sleep:idle --mode=block --who=DL-LAB09 \
        --why='Entrenamiento y entrega del laboratorio 9' \
        env LAB09_INHIBITED=1 bash "$0" "$@"
fi
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u -m src.run_cisco --stage all
.venv/bin/python scripts/build_notebooks.py
.venv/bin/python scripts/execute_notebooks.py
echo 'Entrenamientos, verificación, notebooks e informe terminados.'
