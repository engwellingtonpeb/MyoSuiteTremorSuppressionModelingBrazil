"""Utilitário comum dos scripts: coloca wrist_control/ no path e executa um
módulo dentro da pasta de resultados do estudo (onde ele lê e grava arquivos)."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "wrist_control"
RESULTS = ROOT / "results"


def run(module, results_subdir):
    sys.path.insert(0, str(LIB))
    out = RESULTS / results_subdir
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    print(f"[{module}] resultados em: {out}")
    runpy.run_module(module, run_name="__main__")
