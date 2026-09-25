"""
Teste decisivo DMDc x SINDYc: mesmos dados, mesma síntese H-inf (mixsyn, W1/W3 da
tese), mesmo equilíbrio (u_base) e mesma projeção do espaço nulo; compara:
  - fit de validação (replay em malha fechada)
  - rigidez identificada d(qddot)/dq vs. MuJoCo
  - desempenho em malha fechada na planta MyoSuite (regulação, degraus, tremor 5 Hz)
"""
import numpy as np
import mujoco

import pipeline_3dof_closedloop as main_3dof
import synthesis
import sysid
from plant import WristPlantND


class DMDcModel:
    def __init__(self, A, B, d, dt):
        self.A, self.B, self.d, self.dt = A, B, d, float(dt)

    def step(self, x, u):
        return self.A @ x + self.B @ u + self.d


def true_stiffness(plant, q, a):
    K = np.zeros((plant.n_q, plant.n_q))
    for j in range(plant.n_q):
        acc = []
        for sgn in (+1, -1):
            mujoco.mj_resetData(plant.m, plant.d)
            plant.d.qpos[plant.qadr] = q
            plant.d.qpos[plant.qadr[j]] += sgn * 1e-4
            plant.d.act[plant.act_idx] = a
            mujoco.mj_forward(plant.m, plant.d)
            acc.append(plant.d.qacc[plant.vadr].copy())
        K[:, j] = (acc[0] - acc[1]) / 2e-4
    return K


if __name__ == "__main__":
    import time
    tic = time.perf_counter()
    np.set_printoptions(precision=1, suppress=True, linewidth=140)
    plant = WristPlantND("3dof")
    data = dict(np.load("idtf_data_3dof.npz"))
    sind = sysid.SINDYcModel.load("sindyc_model_3dof.npz")
    dm = sysid.identify(data)
    dmdc = DMDcModel(dm["A"], dm["B"], dm["d"], data["dt"])
    ntr = int(0.7 * len(data["U"]))

    for name, mdl in [("DMDc", dmdc), ("SINDYc", sind)]:
        f = sysid.fit_pct(sysid.replay_closed_loop(mdl.step, data, ntr), data["X"][ntr:])
        x_eq, u_eq, _ = synthesis.equilibrium_nl(mdl.step, 8, 3, data["q_op"], u_base=main_3dof.U_BASE)
        A, _ = synthesis.linearize(mdl.step, x_eq, u_eq)
        Km = A[np.ix_(plant.iv, plant.iq)] / plant.dt
        Kt = true_stiffness(plant, data["q_op"], x_eq[:8])
        err = np.linalg.norm(Km - Kt) / np.linalg.norm(Kt) * 100
        nterms = np.count_nonzero(dm["A"]) + np.count_nonzero(dm["B"]) + 14 if name == "DMDc" else sind.nnz
        print(f"\n########## {name}: {nterms} coeficientes | fit val mín/méd = {f.min():.1f}/{f.mean():.1f} % | "
              f"erro rigidez vs MuJoCo = {err:.0f} %")
        main_3dof.run_control_stage(plant, data, mdl, tic, tag=f"_{name.lower()}")
