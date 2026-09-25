"""[ATUAL] Punho 3 GDL / 8 músculos, palma voltada para o solo (theta = psi = 0°, phi = 80°):
 1) identificação SINDYc em MALHA ABERTA  2) síntese H∞  3) rastreamento  4) tremor de 5 Hz.
Opção: --control-only (reusa o modelo identificado salvo).

Resultados: results/wrist3dof_openloop_palmdown/
"""
from _runner import run

if __name__ == "__main__":
    run("pipeline_3dof_openloop", "wrist3dof_openloop_palmdown")
