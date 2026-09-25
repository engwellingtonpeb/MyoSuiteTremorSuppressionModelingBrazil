"""
Pipeline completo, nativo em Python (sem MATLAB/OpenSim):
  1) experimento de identificação na planta MyoSuite  -> idtf_data.npz
  2) DMDc                                              -> dmdc_model.npz
  3) síntese H-inf (mixsyn) + discretização            -> controller_hinf.npz
  4) malha fechada (sem Matsuoka) na planta não linear -> figuras .png
"""
import time

import numpy as np

import closed_loop
import synthesis
import sysid
from plant import STATE_NAMES

W1 = (10, (30, 1.0), 0.01)    # makeweight(dc, [w, mag], hf) — iguais à tese
W3 = (0.01, (30, 0.9), 1.0)
W2 = 1e-2                     # regularização do esforço (exigência do slycot)

if __name__ == "__main__":
    tic = time.perf_counter()

    print("[1/4] experimento de identificação ...")
    data = sysid.run_identification_experiment(T=40.0, seed=0)
    np.savez("idtf_data.npz", **data)

    print("[2/4] DMDc ...")
    model = sysid.identify(data)
    print("      fit validação [%]:", dict(zip(STATE_NAMES, np.round(model["fit"], 1))))
    np.savez("dmdc_model.npz", **{k: model[k] for k in ["A", "B", "d", "dt", "rmse", "fit"]})
    sysid.plot_identification(data, model)

    print("[3/4] síntese H-inf ...")
    res = synthesis.synthesize(model["A"], model["B"], model["dt"], W1, W3, w2=W2)
    print(f"      gamma = {res['gamma']:.3f} | ordem = {res['Ak'].shape[0]} | "
          f"estável (linear) = {res['stable_dt']}")
    if not res["stable_dt"]:
        raise SystemExit("controlador não estabiliza o modelo linear — revise os pesos")
    np.savez("controller_hinf.npz", Ak=res["Ak"], Bk=res["Bk"], Ck=res["Ck"], Dk=res["Dk"],
             Ts=res["Ts"], gamma=res["gamma"], perf_idx=res["perf_idx"])
    synthesis.plot_loop_shapes(res, W1, W3)

    print("[4/4] malha fechada na planta MyoSuite ...")
    r = closed_loop.run(T=10.0)
    closed_loop.plot_run(r, "malha_fechada_setpoint.png", "H∞ + MyoSuite — regulação (θ=0°, φ=20°)")
    r = closed_loop.run(T=10.0, setpoint=closed_loop.steps)
    closed_loop.plot_run(r, "malha_fechada_degraus.png", "H∞ + MyoSuite — degraus de referência")

    print(f"pipeline completo em {time.perf_counter()-tic:.1f} s")
    import matplotlib.pyplot as plt
    plt.show()
