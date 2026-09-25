"""Visualizador da condição inicial (PNG). Opções: --gif (anima o tremor de 5 Hz), --interactive (viewer MuJoCo).
Usa o controlador salvo pelo script 05.

Resultados: results/wrist3dof_openloop_palmdown/
"""
from _runner import run

if __name__ == "__main__":
    run("visualize", "wrist3dof_openloop_palmdown")
