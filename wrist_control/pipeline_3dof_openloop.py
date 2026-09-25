"""
Pipeline 3 GDL (flexão θ, desvio ψ, pro-supinação φ), 8 músculos, nativo em Python.
Ponto de operação: palma voltada para o solo, mão e antebraço paralelos ao solo
(θ = 0°, ψ = 0°, φ = 80°).

  1) Identificação SINDYc da planta em MALHA ABERTA (episódios curtos a partir
     do equilíbrio; excitação multisenoidal de pequena amplitude, 1-15 Hz)
  2) Síntese H-inf (mixsyn, pesos W1/W3 da tese) sobre o modelo identificado
  3) Rastreamento de referências em malha fechada: controlador + planta MyoSuite
  4) Perturbação de tremor de 5 Hz (excitação alternada agonista/antagonista):
     visualização do tremor na planta

  python scripts/05_wrist3dof_openloop_palmdown.py                 (tudo)
  python scripts/05_wrist3dof_openloop_palmdown.py --control-only  (reusa o modelo salvo)

O fluxo anterior (identificação em malha fechada) está em pipeline_3dof_closedloop.py.
"""
import pickle
import sys
import time

import numpy as np

import closed_loop
import synthesis
import sysid
from plant import WristPlantND

FIT_MIN = 80.0
W1 = (10, (30, 1.0), 0.01)    # makeweight(dc, [w, mag], hf) — iguais à tese
W3 = (0.01, (30, 0.9), 1.0)
W2 = 1e-2                     # regularização do esforço (exigência do slycot)
U_BASE = 0.15                 # co-contração basal (mesma na identificação e no controle)
F_TREMOR, A_TREMOR = 5.0, 0.08


def stage1_identification(plant):
    names = plant.state_names
    print("[1] identificação em MALHA ABERTA (episódios a partir do equilíbrio) ...")
    exp = sysid.run_openloop_episodes(plant, n_ep=60, T_ep=1.5, base=U_BASE, mus_amp=0.03)
    q = np.vstack([np.rad2deg(x[:, plant.iq] - exp["q_op"]) for x in exp["Xs"]])
    lens = np.array([len(x) - 1 for x in exp["Xs"]]) * plant.dt
    print(f"    equilíbrio: q* = {np.rad2deg(exp['x_eq'][plant.iq]).round(2)}°, "
          f"u* = {dict(zip(plant.muscles, exp['u_eq'].round(3)))}")
    print(f"    {len(lens)} episódios, {lens.sum():.0f} s de dados; RMS dq = {q.std(0).round(2)}°, "
          f"max |dq| = {np.abs(q).max(0).round(1)}°; interrompidos pela deriva: {(lens < 1.5).sum()}")
    with open("idtf_malha_aberta.pkl", "wb") as f:
        pickle.dump(exp, f)

    print("[1a] DMDc (referência, mesmos episódios) ...")
    dm = sysid.dmdc_openloop(exp)
    print(f"    fit val min/méd = {dm['fit'].min():.1f} / {dm['fit'].mean():.1f} %")

    print(f"[1b] SINDYc — esparsificando até min(fit) >= {FIT_MIN:.0f}% ...")
    mdl = sysid.identify_sindyc_openloop(exp, plant.n_mus, plant.n_q, fit_min=FIT_MIN)
    print("    fit val [%]:", {k: round(float(v), 1) for k, v in zip(names, mdl.fit)})
    mdl.save("sindyc_malha_aberta.npz")
    sysid.plot_openloop_identification(exp, mdl, dm, names)
    mnames = sysid.lib_mech_names(plant.coords, plant.muscles, trig=False)
    for j, st in enumerate(names[plant.n_mus:]):
        nz = np.nonzero(mdl.Xi_mech[:, j])[0]
        print(f"    Δ{st:9s}: {len(nz):3d} termos -> " + ", ".join(mnames[k] for k in nz[:10])
              + (" ..." if len(nz) > 10 else ""))
    return exp, mdl


def stage2_synthesis(plant, exp, mdl):
    print("[2] síntese H-inf sobre o modelo SINDYc linearizado no ponto de operação ...")
    na, nq = plant.n_mus, plant.n_q
    q_op = exp["q_op"]
    x_eq, u_eq, resid = synthesis.equilibrium_nl(mdl.step, na, nq, q_op, u_base=U_BASE)
    A, B = synthesis.linearize(mdl.step, x_eq, u_eq)
    ev = np.linalg.eigvals(synthesis.d2c_zoh(A, B, mdl.dt)[0])
    print(f"    u* (modelo) = {u_eq.round(3)}  (resíduo {resid:.1e})")
    print(f"    polos instáveis do modelo (malha aberta): {np.round(ev[ev.real > 1e-3].real, 2)} 1/s")
    res = synthesis.synthesize(A, B, mdl.dt, W1, W3, w2=W2, perf_idx=plant.iq)
    print(f"    γ = {res['gamma']:.3f} | ordem = {res['Ak'].shape[0]} | "
          f"estável c/ o modelo = {res['stable_dt']}")
    # pré-alimentação u*(r) linearizada em torno de q_op (evita resolver o equilíbrio a cada passo)
    J = np.zeros((na, nq))
    for j in range(nq):
        dq = np.zeros(nq)
        dq[j] = np.deg2rad(1.0)
        up = synthesis.equilibrium_nl(mdl.step, na, nq, q_op + dq, u_base=U_BASE)[1]
        um = synthesis.equilibrium_nl(mdl.step, na, nq, q_op - dq, u_base=U_BASE)[1]
        J[:, j] = (up - um) / (2 * dq[j])
    Tm = A[np.ix_(plant.iv, np.arange(na))] / mdl.dt          # d(qddot)/d(a)
    ctrl = dict(Ak=res["Ak"], Bk=res["Bk"], Ck=res["Ck"], Dk=res["Dk"], Ts=res["Ts"],
                gamma=res["gamma"], x_eq=x_eq, u_eq=u_eq, J_ff=J, torque_map=Tm,
                q_op=q_op, u_base=U_BASE)
    np.savez("controller_hinf_3dof.npz", **ctrl)
    synthesis.plot_loop_shapes(res, W1, W3, fname="sintese_hinf_3dof.png")
    return ctrl


def make_controller(plant, ctrl):
    ff = lambda r: np.clip(ctrl["u_eq"] + ctrl["J_ff"] @ (np.asarray(r) - ctrl["q_op"]), 0, 1)
    return closed_loop.ControllerND(ctrl, ff, plant.iq, torque_map=ctrl["torque_map"])


def rmse(plant, r, t_from=0.3):
    k = r["t"][:-1] >= t_from
    return np.sqrt(np.mean(np.rad2deg(r["R"][k] - r["X"][:-1][k][:, plant.iq]) ** 2, 0))


def stage3_tracking(plant, ctrl):
    print("[3] rastreamento de referências (controlador + planta MyoSuite) ...")
    q_op = ctrl["q_op"]
    # φ mantido em 77-83°: o tendão do SUP no myoArm troca de lado (wrapping) em φ = 75°
    tab = [(0, 0, 0, 0), (1, 5, 0, 0), (2, 5, 3, 0), (3, 5, 3, -3), (4, -5, -3, -3),
           (5, -5, 0, 3), (6, 0, 0, 0)]

    def steps(t):
        for t0, a, b, c in reversed(tab):
            if t >= t0:
                return q_op + np.deg2rad([a, b, c])

    r1 = closed_loop.run_nd(plant, make_controller(plant, ctrl), 7.0, steps, q0=q_op)
    print(f"    degraus (7 s em {r1['wall']:.1f} s): RMSE [θ ψ φ] = {rmse(plant, r1).round(2)}°")
    closed_loop.plot_run_nd(r1, plant, "rastreamento_degraus.png",
                            "Rastreamento — degraus (palma para baixo, φ₀ = 80°)")

    def sines(t):   # φ em 77-83° (acima do salto do SUP em 75°, abaixo do limite de 90°)
        return q_op + np.deg2rad([6 * np.sin(2 * np.pi * 0.5 * t),
                                  3 * np.sin(2 * np.pi * 0.3 * t),
                                  3 * np.sin(2 * np.pi * 0.4 * t)])

    r2 = closed_loop.run_nd(plant, make_controller(plant, ctrl), 8.0, sines, q0=q_op)
    print(f"    senoides lentas (8 s em {r2['wall']:.1f} s): RMSE [θ ψ φ] = "
          f"{rmse(plant, r2, 1.0).round(2)}°")
    closed_loop.plot_run_nd(r2, plant, "rastreamento_senoides.png",
                            "Rastreamento — senoides 0,3–0,5 Hz (palma para baixo)")
    return r1, r2


def stage4_tremor(plant, ctrl, T=6.0, t_on=1.0):
    print(f"[4] perturbação de tremor {F_TREMOR:.0f} Hz na planta (a partir de t = {t_on} s) ...")
    q_op = ctrl["q_op"]

    def trem(k, x):
        tk = k * plant.dt
        if tk < t_on:
            return np.zeros(plant.n_mus)
        return sysid.tremor_excitation(plant, tk, F_TREMOR, A_TREMOR)

    r = closed_loop.run_nd(plant, make_controller(plant, ctrl), T, lambda t: q_op, q0=q_op,
                           disturbance=trem)
    k = r["t"][:-1] >= t_on + 1.0
    q = np.rad2deg(r["X"][:-1][k][:, plant.iq])
    f = np.fft.rfftfreq(len(q), plant.dt)
    P = np.abs(np.fft.rfft(q - q.mean(0), axis=0)) ** 2
    i1 = np.argmax(f > 1)
    fpk = f[np.argmax(P[i1:], axis=0) + i1]
    print(f"    tremor na planta: pico-a-pico [θ ψ φ] = {(q.max(0) - q.min(0)).round(2)}°, "
          f"RMS = {q.std(0).round(2)}°, frequência dominante = {fpk.round(2)} Hz")
    plot_tremor(r, plant, t_on, f, P)
    with open("tremor_run.pkl", "wb") as fh:
        pickle.dump(r, fh)
    return r


def plot_tremor(r, plant, t_on, f, P):
    import matplotlib.pyplot as plt
    t, X = r["t"], r["X"]
    labels = ["θ flexão", "ψ desvio rad./uln.", "φ pro-sup"]
    fig, axs = plt.subplots(3, 3, figsize=(15, 9))
    for j in range(3):
        ax = axs[0, j]
        ax.plot(t, np.rad2deg(X[:, plant.iq[j]]), "k", lw=0.9)
        ax.axvline(t_on, color="r", ls=":", label="tremor ligado")
        ax.set_title(labels[j]); ax.set_ylabel("[°]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax = axs[1, j]
        sl = (t >= t_on + 2.0) & (t <= t_on + 3.0)
        ax.plot(t[sl], np.rad2deg(X[sl, plant.iq[j]]), "k")
        ax.set_title(labels[j] + " — zoom de 1 s"); ax.grid(alpha=0.3); ax.set_xlabel("tempo [s]")
        ax = axs[2, j]
        ax.semilogy(f, P[:, j] / P[:, j].max(), "b")
        ax.set_xlim(0, 20); ax.axvline(F_TREMOR, color="r", ls=":")
        ax.set_title("espectro (normalizado)"); ax.set_xlabel("frequência [Hz]"); ax.grid(alpha=0.3)
    fig.suptitle(f"Tremor de {F_TREMOR:.0f} Hz na planta MyoSuite (controlador H∞ mantendo a postura)")
    fig.tight_layout()
    fig.savefig("tremor_5Hz_planta.png", dpi=130)


if __name__ == "__main__":
    tic = time.perf_counter()
    plant = WristPlantND("3dof")
    if "--control-only" in sys.argv:
        with open("idtf_malha_aberta.pkl", "rb") as fh:
            exp = pickle.load(fh)
        mdl = sysid.SINDYcModel.load("sindyc_malha_aberta.npz")
    else:
        exp, mdl = stage1_identification(plant)
    ctrl = stage2_synthesis(plant, exp, mdl)
    stage3_tracking(plant, ctrl)
    stage4_tremor(plant, ctrl)
    print(f"pipeline completo em {time.perf_counter() - tic:.0f} s")
