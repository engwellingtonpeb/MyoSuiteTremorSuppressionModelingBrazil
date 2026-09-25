"""
Malha de controle motor do punho (sem oscilador de Matsuoka) no MyoSuite.
Porta para Python o OsimControlsFcn.m / OsimPlantFcn.m (FES = 'off'):

    e   = r - x,   x = [asup aecrl afcu apq theta phi thetadot phidot]
    xk+ = Ak xk + Bk e ;  du = Ck xk + Dk e        (H-inf discreto, Ts=1 ms)
    u   = u* + du                                  (u* = equilíbrio do modelo)
    [opcional] inibição recíproca: u_i <- k_i * alpha_i(eps) * u_i
    saturação 0 <= u <= 1 ;  u7 = [SUP ECRL ECRB ECU FCR FCU PQ], demais = 0.01
"""
import time

import numpy as np

from plant import WristPlant, STATE_NAMES
from synthesis import equilibrium


class InnerLoopController:
    def __init__(self, ctrl, A, B, d, reciprocal_inhibition=False,
                 k=(1, 1, 1, 1), t_init=0.1, u_init=(0.1, 0.01, 0.1, 0.01)):
        self.Ak, self.Bk, self.Ck, self.Dk = ctrl["Ak"], ctrl["Bk"], ctrl["Ck"], ctrl["Dk"]
        self.perf_idx = ctrl["perf_idx"]
        self.A, self.B, self.d = A, B, d
        self.ri = reciprocal_inhibition
        self.k = np.asarray(k, float)
        self.t_init = t_init
        self.u_init = np.asarray(u_init)
        self.reset()

    def reset(self):
        self.xk = np.zeros(self.Ak.shape[0])
        self._sp = None

    def set_setpoint(self, theta_ref, phi_ref):
        if self._sp == (theta_ref, phi_ref):
            return
        self._sp = (theta_ref, phi_ref)
        self.x_eq, self.u_eq = equilibrium(self.A, self.B, self.d, theta_ref, phi_ref)
        self.r = self.x_eq.copy()          # r = [a*(4) theta_ref phi_ref 0 0]

    def __call__(self, t, x):
        e = (self.r - x)[self.perf_idx]
        du = self.Ck @ self.xk + self.Dk @ e
        self.xk = self.Ak @ self.xk + self.Bk @ e
        u = self.u_eq + du

        if self.ri:  # inibição recíproca (tanh do erro em graus), como na tese
            eps_theta, eps_phi = np.rad2deg(self.r[4] - x[4]), np.rad2deg(self.r[5] - x[5])
            alpha = np.array([0.5 - 0.5 * np.tanh(eps_theta),   # ECRL
                              0.5 + 0.5 * np.tanh(eps_theta),   # FCU
                              0.5 + 0.5 * np.tanh(eps_phi),   # PQ
                              0.5 - 0.5 * np.tanh(eps_phi)])  # SUP
            u = self.k * alpha * u
        if t < self.t_init:                # inicialização do modelo
            u = self.u_init.copy()
        return np.clip(u, 0.0, 1.0)        # [ECRL FCU PQ SUP]


def run(T=10.0, setpoint=lambda t: (0.0, np.deg2rad(20)), x0=(-0.1745, 0.5236),
        reciprocal_inhibition=False, verbose=True):
    mdl = np.load("dmdc_model.npz")
    ctrl = dict(np.load("controller_hinf.npz"))
    plant = WristPlant()
    C = InnerLoopController(ctrl, mdl["A"], mdl["B"], mdl["d"],
                            reciprocal_inhibition=reciprocal_inhibition)
    N = int(round(T / plant.dt))
    t = np.arange(N + 1) * plant.dt
    X = np.zeros((N + 1, 8)); U = np.zeros((N, 4)); R = np.zeros((N, 2))
    X[0] = plant.reset(*x0)

    t0 = time.perf_counter()
    for k in range(N):
        R[k] = setpoint(t[k])
        C.set_setpoint(*R[k])
        U[k] = C(t[k], X[k])
        X[k + 1] = plant.step(plant.excitation_vector(U[k]))
    wall = time.perf_counter() - t0
    if verbose:
        print(f"{T:.0f} s simulados em {wall:.2f} s  (fator tempo real {T/wall:.1f}x)")
    return dict(t=t, X=X, U=U, R=R, wall=wall)


def plot_run(res, fname, title):
    import matplotlib.pyplot as plt
    t, X, U, R = res["t"], res["X"], res["U"], res["R"]
    fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j, (i, nome) in enumerate([(4, "θ  flexão"), (5, "φ  pro-supinação")]):
        ax = axs[0, j]
        ax.plot(t, np.rad2deg(X[:, i]), "k", label="MyoSuite")
        ax.plot(t[:-1], np.rad2deg(R[:, j]), "r--", label="referência")
        ax.set_title(nome); ax.set_ylabel("[°]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axs[1, j].plot(t, np.rad2deg(X[:, i + 2]), "k")
        axs[1, j].set_ylabel("[°/s]"); axs[1, j].grid(alpha=0.3)
        axs[1, j].set_title(f"{nome.split()[0]}̇")
    for lab, iu, ia in [("ECRL", 0, 1), ("FCU", 1, 2)]:
        l, = axs[2, 0].plot(t[:-1], U[:, iu], "--", lw=0.8)
        axs[2, 0].plot(t, X[:, ia], color=l.get_color(), label=lab)
    for lab, iu, ia in [("PQ", 2, 3), ("SUP", 3, 0)]:
        l, = axs[2, 1].plot(t[:-1], U[:, iu], "--", lw=0.8)
        axs[2, 1].plot(t, X[:, ia], color=l.get_color(), label=lab)
    for ax in axs[2]:
        ax.set_title("excitação (--) e ativação (—)"); ax.set_xlabel("tempo [s]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)


# =====================================================================
#  N GDL (3 GDL / 8 músculos) — controlador sobre o modelo SINDYc
# =====================================================================
class ControllerND:
    """
    du = K(z) (r - q) ;  u = u*(r) + du ;  0 <= u <= 1
    u*(r): equilíbrio do modelo SINDYc para o setpoint (calculado uma vez por setpoint).
    """

    def __init__(self, ctrl, eq_fn, iq, torque_map=None):
        """
        torque_map: matriz (n_q x n_mus) d(qddot)/d(a) do modelo. Se dada, o
        comando du é projetado fora do seu espaço nulo (combinações musculares
        que só alteram a co-contração, invisíveis às saídas) — evita a deriva
        lenta da co-contração até a saturação em 0 sem alterar a malha linear.
        """
        self.Ak, self.Bk, self.Ck, self.Dk = ctrl["Ak"], ctrl["Bk"], ctrl["Ck"], ctrl["Dk"]
        self.eq_fn, self.iq = eq_fn, np.asarray(iq)
        self.xk = np.zeros(self.Ak.shape[0])
        self._sp, self._cache = None, {}
        self.P = None
        if torque_map is not None:
            _, s, Vt = np.linalg.svd(torque_map)
            N = Vt[len(s):].T                      # base do espaço nulo (n_mus x n_mus-n_q)
            self.P = np.eye(N.shape[0]) - N @ N.T

    def set_setpoint(self, q_ref):
        key = tuple(np.round(q_ref, 6))
        if key == self._sp:
            return
        self._sp = key
        if key not in self._cache:
            self._cache[key] = self.eq_fn(np.asarray(q_ref))
        self.r, self.u_eq = np.asarray(q_ref), self._cache[key]

    def __call__(self, x):
        e = self.r - x[self.iq]
        du = self.Ck @ self.xk + self.Dk @ e
        self.xk = self.Ak @ self.xk + self.Bk @ e
        if self.P is not None:
            du = self.P @ du
        return np.clip(self.u_eq + du, 0.0, 1.0)


def run_nd(plant, controller, T, setpoint, q0, disturbance=None):
    """disturbance(k, x) -> excitação aditiva (n_mus), p.ex. um sinal de tremor."""
    N = int(round(T / plant.dt))
    t = np.arange(N + 1) * plant.dt
    X = np.zeros((N + 1, len(plant.state_names)))
    U = np.zeros((N, plant.n_mus))
    R = np.zeros((N, plant.n_q))
    X[0] = plant.reset(q0=q0, a0=0.05)
    t0 = time.perf_counter()
    for k in range(N):
        R[k] = setpoint(t[k])
        controller.set_setpoint(R[k])
        u = controller(X[k])
        if disturbance is not None:
            u = u + disturbance(k, X[k])
        U[k] = np.clip(u, 0, 1)
        X[k + 1] = plant.step(U[k])
    return dict(t=t, X=X, U=U, R=R, wall=time.perf_counter() - t0)


def plot_run_nd(res, plant, fname, title):
    import matplotlib.pyplot as plt
    t, X, R = res["t"], res["X"], res["R"]
    nq = plant.n_q
    fig, axs = plt.subplots(3, nq, figsize=(5 * nq, 9), sharex=True)
    labels = {"theta": "θ flexão", "psi": "ψ desvio rad./uln.", "phi": "φ pro-sup"}
    groups = [["ECRL", "ECRB", "FCR", "FCU"], ["ECU", "FCU", "ECRL", "FCR"], ["SUP", "PQ", "PT"]]
    for j, c in enumerate(plant.coords):
        ax = axs[0, j]
        ax.plot(t, np.rad2deg(X[:, plant.iq[j]]), "k", label="MyoSuite")
        ax.plot(t[:-1], np.rad2deg(R[:, j]), "r--", label="referência")
        ax.set_title(labels.get(c, c)); ax.set_ylabel("[°]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axs[1, j].plot(t, np.rad2deg(X[:, plant.iv[j]]), "k")
        axs[1, j].set_ylabel("[°/s]"); axs[1, j].grid(alpha=0.3)
        axs[1, j].set_title(labels.get(c, c) + " — velocidade")
        for mname in groups[j]:
            axs[2, j].plot(t, X[:, plant.muscles.index(mname)], label=mname)
        axs[2, j].set_title("ativações"); axs[2, j].set_xlabel("tempo [s]")
        axs[2, j].grid(alpha=0.3); axs[2, j].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)


def steps(t):
    tab = [(0, 0, 20), (2, 10, 20), (4, -10, 20), (6, 0, 30), (8, 5, 10)]
    for t0, ph, ps in reversed(tab):
        if t >= t0:
            return np.deg2rad(ph), np.deg2rad(ps)


if __name__ == "__main__":
    def metrics(res, t_from=1.0):
        k = res["t"][:-1] >= t_from
        e = np.rad2deg(res["R"][k] - res["X"][:-1][k][:, 4:6])
        return np.sqrt(np.mean(e ** 2, 0))

    r1 = run()
    print("  RMSE (t>1s) [theta phi] em graus:", np.round(metrics(r1), 3),
          " | final:", np.round(np.rad2deg(r1["X"][-1, 4:6]), 2))
    plot_run(r1, "malha_fechada_setpoint.png", "H∞ + MyoSuite — regulação (θ=0°, φ=20°)")

    r2 = run(setpoint=steps)
    print("  RMSE (t>1s) [theta phi] em graus:", np.round(metrics(r2), 3))
    plot_run(r2, "malha_fechada_degraus.png", "H∞ + MyoSuite — degraus de referência")

    r3 = run(setpoint=steps, reciprocal_inhibition=True)
    print("  [c/ inibição recíproca] RMSE [theta phi]:", np.round(metrics(r3), 3))
    plot_run(r3, "malha_fechada_degraus_RI.png", "H∞ + inibição recíproca — degraus")
