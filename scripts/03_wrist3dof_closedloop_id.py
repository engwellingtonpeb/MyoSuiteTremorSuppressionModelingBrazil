"""[HISTÓRICO] Punho 3 GDL / 8 músculos, phi = 20° (palma vertical), identificação SINDYc em MALHA FECHADA.
Opções: --control-only, --ident-only, --compare (estudo de co-contração e projeção do espaço nulo).

Resultados: results/wrist3dof_closedloop_id/
"""
from _runner import run

if __name__ == "__main__":
    run("pipeline_3dof_closedloop", "wrist3dof_closedloop_id")
