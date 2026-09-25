"""
Pipeline 3 GDL (flexão, desvio, pro-supinação) com 8 músculos, nativo em Python:
  1) experimento de PEQUENA amplitude (~2° RMS, banda 1-15 Hz) no ponto de operação
  2) SINDYc esparsificado até min(fit de validação) >= 80 %  (+ DMDc de referência)
  3) equilíbrio e linearização do SINDYc -> síntese H-inf (mixsyn) nos 3 ângulos
  4) malha fechada na planta MyoSuite não linear
"""
import sys
import time

import numpy as np

import closed_loop
import sysid
import synthesis
from plant import WristPlantND

FIT_MIN = 80.0
W1 = (10, (30, 1.0), 0.01)    # makeweight(dc, [w, mag], hf) — iguais à tese
W3 = (0.01, (30, 0.9), 1.0)
W2 = 1e-2                     # regularização do esforço (exigência do slycot)
U_BASE = 0.15                 # co-contração basal no equilíbrio (margem p/ antagonistas)
PROJECT_NULLSPACE = True      # remove de du as combinações que só mudam a co-contração


def run_identification_stage(plant, tic):
    names = plant.state_names
    print("[1] experimento de pequena amplitude (60 s) ...")
    data = sysid.run_tremor_band_experiment(plant, seed=0)
    q = np.rad2deg(data["X"][2000:, plant.iq])
    print(f"    RMS ângulos [theta psi phi] = {q.std(0).round(2)}°, pico = "
          f"{np.abs(q - q.mean(0)).max(0).round(1)}°")
    np.savez("idtf_data_3dof.npz", **data)

    print("[2a] DMDc (referência) ...")
    dm = sysid.identify(data)
    ntr = int(0.7 * len(data["U"]))
    dm["xval"] = sysid.replay_closed_loop(sysid.dmdc_step_fn(dm["A"], dm["B"], dm["d"]), data, ntr)
    dm["fit"] = sysid.fit_pct(dm["xval"], data["X"][ntr:])
    print("    fit val (malha fechada) [%]:", dict(zip(names, np.round(dm["fit"], 1))))

    print(f"[2b] SINDYc — esparsificando até min(fit) >= {FIT_MIN:.0f}% ...")
    mdl, log = sysid.identify_sindyc(data, plant.n_mus, plant.n_q, fit_min=FIT_MIN,
                                     q_ref=data["q_op"], trig=False)
    p_full = 10 * plant.n_mus + sysid.lib_mech(np.zeros((1, 3)), np.zeros((1, 3)),
                                               np.zeros((1, 8)), trig=False).shape[1] * 2 * plant.n_q
    print(f"    termos ativos: {mdl.nnz} de {p_full} candidatos")
    print("    fit val [%]:", dict(zip(names, np.round(mdl.fit, 1))))
    mdl.save("sindyc_model_3dof.npz")
    sysid.plot_sindyc_3dof(data, mdl, dm, names)

    # termos da dinâmica mecânica selecionados
    mnames = sysid.lib_mech_names(plant.coords, plant.muscles, trig=False)
    for j, st in enumerate(names[plant.n_mus:]):
        nz = np.nonzero(mdl.Xi_mech[:, j])[0]
        print(f"    Δ{st:9s}: {len(nz):3d} termos  ->", ", ".join(mnames[k] for k in nz[:12]),
              "..." if len(nz) > 12 else "")
    print(f"tempo até aqui: {time.perf_counter() - tic:.0f} s")
    return data, mdl


def run_control_stage(plant, data, mdl, tic, u_base=U_BASE, project=PROJECT_NULLSPACE,
                      tag=""):
    print(f"[3] equilíbrio (u_base={u_base}) + linearização do SINDYc + síntese H-inf ...")
    na, nq = plant.n_mus, plant.n_q
    x_eq, u_eq, resid = synthesis.equilibrium_nl(mdl.step, na, nq, data["q_op"], u_base=u_base)
    print(f"    u* = {dict(zip(plant.muscles, np.round(u_eq, 3)))}  (resíduo {resid:.1e})")
    A, B = synthesis.linearize(mdl.step, x_eq, u_eq)
    print("    |autovalores| de A (6 maiores):",
          np.round(np.sort(np.abs(np.linalg.eigvals(A)))[-6:], 5))
    res = synthesis.synthesize(A, B, mdl.dt, W1, W3, w2=W2, perf_idx=plant.iq)
    print(f"    gamma = {res['gamma']:.3f} | ordem = {res['Ak'].shape[0]} | modos rápidos "
          f"removidos = {res['n_fast_removed']} | estável (linear) = {res['stable_dt']}")
    Tm = A[np.ix_(plant.iv, np.arange(na))] / mdl.dt if project else None   # d(qddot)/d(a)
    np.savez(f"controller_hinf_3dof{tag}.npz", Ak=res["Ak"], Bk=res["Bk"], Ck=res["Ck"],
             Dk=res["Dk"], Ts=res["Ts"], gamma=res["gamma"], perf_idx=res["perf_idx"],
             x_eq=x_eq, u_eq=u_eq, torque_map=Tm if Tm is not None else np.nan, u_base=u_base)
    synthesis.plot_loop_shapes(res, W1, W3, fname=f"sintese_hinf_3dof{tag}.png")

    print("[4] malha fechada na planta MyoSuite ...")

    def eq_fn(q_ref):
        return synthesis.equilibrium_nl(mdl.step, na, nq, q_ref, u_base=u_base)[1]

    def rmse(r, t_from=0.5):
        k = r["t"][:-1] >= t_from
        return np.sqrt(np.mean(np.rad2deg(r["R"][k] - r["X"][:-1][k][:, plant.iq]) ** 2, 0))

    q_op = data["q_op"]
    C = closed_loop.ControllerND(res, eq_fn, plant.iq, torque_map=Tm)
    r1 = closed_loop.run_nd(plant, C, 5.0, lambda t: q_op, q0=q_op + np.deg2rad([5, -3, 5]))
    print(f"    regulação (5 s em {r1['wall']:.2f} s): RMSE(t>0,5 s) [θ ψ φ] = {rmse(r1).round(2)}°,"
          f" final = {np.rad2deg(r1['X'][-1, plant.iq]).round(2)}°")
    closed_loop.plot_run_nd(r1, plant, f"malha_fechada_3dof_regulacao{tag}.png",
                            "H∞ (SINDYc) 3 GDL — regulação no ponto de operação")

    tab = [(0, 0, 0, 0), (1, 4, 0, 0), (2, 4, 3, 0), (3, 4, 3, -4), (4, -3, -2, 3), (5, 0, 0, 0)]

    def sp(t):
        for t0, a, b, c in reversed(tab):
            if t >= t0:
                return q_op + np.deg2rad([a, b, c])

    C = closed_loop.ControllerND(res, eq_fn, plant.iq, torque_map=Tm)
    r2 = closed_loop.run_nd(plant, C, 6.0, sp, q0=q_op)
    print(f"    degraus pequenos (6 s em {r2['wall']:.2f} s): RMSE [θ ψ φ] = {rmse(r2).round(2)}°")
    closed_loop.plot_run_nd(r2, plant, f"malha_fechada_3dof_degraus{tag}.png",
                            "H∞ (SINDYc) 3 GDL — degraus de pequena amplitude")

    # rejeição de perturbação na banda do tremor: torque senoidal de 5 Hz por GDL
    Rn = data["ctrl_Rn"]

    def tremor(k, x):
        tau = 0.6 * np.sin(2 * np.pi * 5.0 * k * plant.dt + np.array([0.0, 2.1, 4.2]))
        return np.maximum(Rn.T @ tau, 0)

    ctrl_post = {k[5:]: v for k, v in data.items() if k.startswith("ctrl_")}

    class Postural:   # controle 'voluntário' PD+I usado na identificação (referência)
        def __init__(self):
            self.integ = np.zeros(nq)

        def set_setpoint(self, r):
            pass

        def __call__(self, x):
            u, self.integ = sysid.postural_command(ctrl_post, x, self.integ, None)
            return u

    r_vol = closed_loop.run_nd(plant, Postural(), 5.0, lambda t: q_op, q0=q_op, disturbance=tremor)
    C = closed_loop.ControllerND(res, eq_fn, plant.iq, torque_map=Tm)
    r_hinf = closed_loop.run_nd(plant, C, 5.0, lambda t: q_op, q0=q_op, disturbance=tremor)
    a_vol, a_hinf = rmse(r_vol, 1.0), rmse(r_hinf, 1.0)
    print(f"    tremor 5 Hz — RMS [θ ψ φ]: PD+I voluntário = {a_vol.round(2)}°, "
          f"H∞ = {a_hinf.round(2)}°  (redução {(100 * (1 - a_hinf / a_vol)).round(0)} %)")
    closed_loop.plot_run_nd(r_hinf, plant, f"malha_fechada_3dof_tremor{tag}.png",
                            "H∞ (SINDYc) 3 GDL — perturbação de 5 Hz (tremor)")
    print(f"pipeline completo em {time.perf_counter() - tic:.0f} s")


if __name__ == "__main__":
    tic = time.perf_counter()
    plant = WristPlantND("3dof")
    if "--control-only" in sys.argv:
        data = dict(np.load("idtf_data_3dof.npz"))
        mdl = sysid.SINDYcModel.load("sindyc_model_3dof.npz")
    else:
        data, mdl = run_identification_stage(plant, tic)
    if "--compare" in sys.argv:          # estudo: co-contração basal x projeção
        for ub, pj in [(0.05, False), (0.05, True), (0.10, False), (0.10, True), (0.15, True)]:
            print(f"\n===== u_base={ub}  projeção={pj} =====")
            run_control_stage(plant, data, mdl, tic, u_base=ub, project=pj,
                              tag=f"_ub{int(ub * 100):02d}{'_P' if pj else ''}")
    elif "--ident-only" not in sys.argv:
        run_control_stage(plant, data, mdl, tic)
