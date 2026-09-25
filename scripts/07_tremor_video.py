"""Vídeo do tremor básico (perturbação senoidal de 5 Hz, 10 s) nos 3 planos anatômicos
(sagital, frontal, transversal), com o controlador H∞ sustentando a postura de palma
para baixo. Usa o controlador salvo pelo script 05.

Resultados: results/wrist3dof_openloop_palmdown/video/
"""
import shutil

from _runner import RESULTS, run

if __name__ == "__main__":
    src = RESULTS / "wrist3dof_openloop_palmdown" / "controller_hinf_3dof.npz"
    dst = RESULTS / "wrist3dof_openloop_palmdown" / "video"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst / src.name)
    run("tremor_video", "wrist3dof_openloop_palmdown/video")
