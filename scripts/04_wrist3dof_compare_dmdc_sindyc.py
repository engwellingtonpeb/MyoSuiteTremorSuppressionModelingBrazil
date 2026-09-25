"""[HISTÓRICO] Punho 3 GDL: mesma síntese H∞ a partir do DMDc e do SINDYc (usa os dados do script 03).

Resultados: results/wrist3dof_closedloop_id/
"""
from _runner import run

if __name__ == "__main__":
    run("compare_dmdc_sindyc", "wrist3dof_closedloop_id")
