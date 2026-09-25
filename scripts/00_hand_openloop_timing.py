"""
Simulação em malha aberta da dinâmica direta da mão (MyoHand / MyoSuite).

- Modelo: myoHand (39 músculos, 23 GDL) via ambiente 'myoHandPoseFixed-v0'.
- Entrada: excitações musculares pré-definidas (sem realimentação), num
  padrão cíclico de "abrir/fechar a mão" (flexores x extensores em antifase).
- Saída: ângulos articulares, excitações e ativações musculares por 10 s,
  além da medição do tempo de execução (wall-clock) da simulação.

Duas formas de integração são cronometradas:
  1) env.step()  -> API do MyoSuite (inclui cálculo de observação/recompensa)
  2) mujoco.mj_step() -> dinâmica direta "pura" no mesmo modelo MuJoCo
"""
import time
import warnings

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from myosuite.utils import gym

warnings.filterwarnings("ignore")

T_FINAL = 10.0          # duração do movimento [s]
FREQ = 0.5              # frequência do ciclo abrir/fechar [Hz]
BASE = 0.05             # excitação basal (tônus) dos demais músculos
AMP = 0.6               # amplitude das excitações moduladas

FLEXORES = ["FCR", "FCU", "PL", "FDS2", "FDS3", "FDS4", "FDS5",
            "FDP2", "FDP3", "FDP4", "FDP5", "FPL", "OP"]
EXTENSORES = ["ECRL", "ECRB", "ECU", "EDC2", "EDC3", "EDC4", "EDC5",
              "EDM", "EIP", "EPL", "EPB", "APL"]


def excitacao(t, nomes):
    """Excitação u(t) em [0, 1] para cada músculo (malha aberta)."""
    u = np.full(len(nomes), BASE)
    s = 0.5 * (1 - np.cos(2 * np.pi * FREQ * t))      # 0 -> 1 -> 0
    for i, n in enumerate(nomes):
        if n in FLEXORES:
            u[i] = BASE + AMP * s
        elif n in EXTENSORES:
            u[i] = BASE + AMP * (1 - s)
    return np.clip(u, 0.0, 1.0)


def simula_env():
    """Simulação usando a API do MyoSuite (env.step)."""
    # normalize_act=False -> a ação é diretamente a excitação muscular em [0,1]
    env = gym.make("myoHandPoseFixed-v0", normalize_act=False)
    env.reset(seed=0)
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data
    nomes_mus = [m.actuator(i).name for i in range(m.nu)]
    nomes_jnt = [m.joint(i).name for i in range(m.njnt)]
    n_passos = int(round(T_FINAL / uw.dt))

    t_log = np.zeros(n_passos)
    q_log = np.zeros((n_passos, m.njnt))
    u_log = np.zeros((n_passos, m.nu))
    a_log = np.zeros((n_passos, m.na))

    t0 = time.perf_counter()
    for k in range(n_passos):
        u = excitacao(d.time, nomes_mus)
        env.step(u)
        t_log[k] = d.time
        q_log[k] = d.qpos[m.jnt_qposadr]
        u_log[k] = u          # (o MyoSuite zera d.ctrl após o passo)
        a_log[k] = d.act
    t_exec = time.perf_counter() - t0

    info = dict(dt=uw.dt, timestep=m.opt.timestep, frame_skip=uw.frame_skip,
                n_passos=n_passos, t_exec=t_exec)
    env.close()
    return t_log, q_log, u_log, a_log, nomes_jnt, nomes_mus, info


def simula_mujoco_puro():
    """Mesma dinâmica direta, integrando só com mujoco.mj_step (sem overhead)."""
    env = gym.make("myoHandPoseFixed-v0", normalize_act=False)
    env.reset(seed=0)
    m = env.unwrapped.mj_model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    nomes_mus = [m.actuator(i).name for i in range(m.nu)]
    n_passos = int(round(T_FINAL / m.opt.timestep))

    t0 = time.perf_counter()
    for _ in range(n_passos):
        d.ctrl[:] = excitacao(d.time, nomes_mus)
        mujoco.mj_step(m, d)
    t_exec = time.perf_counter() - t0
    env.close()
    return t_exec, n_passos


def plota(t, q, u, a, nomes_jnt, nomes_mus):
    q_deg = np.rad2deg(q)
    grupos = {
        "Punho / antebraço": ["pro_sup", "deviation", "flexion"],
        "Polegar": ["cmc_abduction", "cmc_flexion", "mp_flexion", "ip_flexion"],
        "Indicador": ["mcp2_flexion", "mcp2_abduction", "pm2_flexion", "md2_flexion"],
        "Médio": ["mcp3_flexion", "mcp3_abduction", "pm3_flexion", "md3_flexion"],
        "Anelar": ["mcp4_flexion", "mcp4_abduction", "pm4_flexion", "md4_flexion"],
        "Mínimo": ["mcp5_flexion", "mcp5_abduction", "pm5_flexion", "md5_flexion"],
    }
    fig1, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for ax, (titulo, jnts) in zip(axs.flat, grupos.items()):
        for j in jnts:
            ax.plot(t, q_deg[:, nomes_jnt.index(j)], label=j)
        ax.set_title(titulo)
        ax.set_ylabel("ângulo [°]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axs[-1]:
        ax.set_xlabel("tempo [s]")
    fig1.suptitle("MyoHand – ângulos articulares (malha aberta)")
    fig1.tight_layout()
    fig1.savefig("angulos_articulares.png", dpi=150)

    fig2, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1, 2]})
    exemplos = ["FDS3", "EDC3"]   # um flexor e um extensor
    for n in exemplos:
        i = nomes_mus.index(n)
        ax1.plot(t, u[:, i], "--", lw=1)
        ax1.plot(t, a[:, i], lw=1.5, label=n, color=ax1.lines[-1].get_color())
    ax1.set_title("Excitação (tracejado) x ativação (contínuo) – músculos selecionados")
    ax1.set_ylabel("[0–1]")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.plot(t, a.mean(axis=1), "k", label="média de todos os músculos")
    ax2.set_ylabel("ativação média")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    im = ax3.imshow(a.T, aspect="auto", origin="lower", cmap="viridis",
                    extent=[t[0], t[-1], -0.5, len(nomes_mus) - 0.5],
                    vmin=0, vmax=1)
    ax3.set_yticks(range(len(nomes_mus)))
    ax3.set_yticklabels(nomes_mus, fontsize=6)
    ax3.set_xlabel("tempo [s]")
    ax3.set_title("Ativações musculares (39 músculos)")
    fig2.colorbar(im, ax=ax3, label="ativação")
    fig2.tight_layout()
    fig2.savefig("ativacoes_musculares.png", dpi=150)


if __name__ == "__main__":
    import os
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "results" / "hand_openloop_timing"
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    t, q, u, a, nomes_jnt, nomes_mus, info = simula_env()
    t_puro, n_puro = simula_mujoco_puro()

    print("\n================ TEMPO DE EXECUÇÃO ================")
    print(f"Tempo simulado          : {T_FINAL:.1f} s")
    print(f"Passo MuJoCo            : {info['timestep']*1e3:.1f} ms "
          f"(frame_skip={info['frame_skip']}, dt env={info['dt']*1e3:.0f} ms)")
    print(f"[MyoSuite env.step]  {info['n_passos']} passos -> "
          f"{info['t_exec']:.3f} s  (fator tempo real: "
          f"{T_FINAL/info['t_exec']:.1f}x)")
    print(f"[mujoco.mj_step]     {n_puro} passos -> "
          f"{t_puro:.3f} s  (fator tempo real: {T_FINAL/t_puro:.1f}x)")
    print("===================================================")

    plota(t, q, u, a, nomes_jnt, nomes_mus)
    print("Figuras salvas: angulos_articulares.png, ativacoes_musculares.png")
    plt.show()
