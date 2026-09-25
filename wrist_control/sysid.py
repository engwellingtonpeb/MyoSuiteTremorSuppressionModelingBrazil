"""
Identificação de sistemas da planta do punho (MyoSuite) por DMDc.

Porta para Python o protocolo de 04_DMDc_IDTF (dmdc_3.m / DelayDMDc_MV.m):
    x(k+1) = A x(k) + B u(k) [+ d]
    x = [asup aecrl afcu apq theta phi thetadot phidot],  u = [uecrl ufcu upq usup]

Experimento de identificação: a planta é conduzida por um controlador
"voluntário" rudimentar (PD antagonista) que segue degraus de referência
(como em 06_utils/idtf_signals.m), somado a sinais PRBS independentes em cada
músculo para garantir excitação persistente. As entradas registradas são as
excitações totais efetivamente aplicadas.
"""
import mujoco
import numpy as np

from plant import WristPlant, STATE_NAMES
from sindyc import stlsq


# ------------------------------------------------------------------ sinais
def prbs(t, taulim, levels, rng, t_end):
    """PRBS multinível com tempos de chaveamento aleatórios (porta de prbs.m)."""
    taus = [0.0]
    while taus[-1] < t_end:
        taus.append(taus[-1] + rng.uniform(*taulim))
    lv = rng.choice(levels, size=len(taus))
    return lv[np.searchsorted(taus, t, side="right") - 1]


def reference_steps(t, rng, T_step=2.0, theta_lim=(-15, 15), phi_lim=(10, 40)):
    """Degraus aleatórios de referência [theta_ref, phi_ref] em rad."""
    n = int(np.ceil(t[-1] / T_step)) + 1
    theta = np.deg2rad(rng.uniform(*theta_lim, n))
    phi = np.deg2rad(rng.uniform(*phi_lim, n))
    k = np.minimum((t // T_step).astype(int), n - 1)
    return np.stack([theta[k], phi[k]], axis=1)


def run_identification_experiment(T=40.0, seed=0, prbs_amp=0.15,
                                  base=0.05, kp=1.5, kd=0.15, plant=None):
    rng = np.random.default_rng(seed)
    p = plant or WristPlant()
    dt = p.dt
    N = int(round(T / dt))
    t = np.arange(N) * dt
    ref = reference_steps(t, rng)
    pr = np.stack([prbs(t, (0.02, 0.2), [0.0, prbs_amp], rng, T) for _ in range(4)], 1)

    X = np.zeros((N + 1, 8))
    U = np.zeros((N, 4))
    X[0] = p.reset(theta0=ref[0, 0], phi0=ref[0, 1])
    for k in range(N):
        x = X[k]
        etheta = x[4] - ref[k, 0]          # >0 -> muito flexionado -> ECRL
        ephi = x[5] - ref[k, 1]          # >0 -> SUP reduz phi
        vtheta = kp * etheta + kd * x[6]
        vphi = kp * ephi + kd * x[7]
        u = np.array([base + max(vtheta, 0),   # ECRL
                      base + max(-vtheta, 0),  # FCU
                      base + max(-vphi, 0),  # PQ
                      base + max(vphi, 0)])  # SUP
        u = np.clip(u + pr[k], 0, 1)
        U[k] = u
        X[k + 1] = p.step(p.excitation_vector(u))
    return dict(t=t, X=X, U=U, ref=ref, dt=dt)


# ------------------------------------------------------------------ DMDc
def dmdc(X, Xp, U, affine=True, tol=1e-10, rank=None):
    """
    DMDc (Proctor, Brunton & Kutz 2016) — porta de DelayDMDc_MV.m (stackmax=1).
    X, Xp: (n, m) snapshots;  U: (q, m) entradas.  Retorna A, B, d.
    """
    n = X.shape[0]
    Gam = np.vstack([U, np.ones((1, U.shape[1]))]) if affine else U
    Omega = np.vstack([X, Gam])
    Uo, S, Vt = np.linalg.svd(Omega, full_matrices=False)
    r = rank or int(np.sum(S > tol))
    G = Xp @ Vt[:r].T @ np.diag(1 / S[:r]) @ Uo[:, :r].T
    A, B = G[:, :n], G[:, n:n + U.shape[0]]
    d = G[:, -1] if affine else np.zeros(n)
    return A, B, d


def simulate_linear(A, B, d, x0, U):
    x = np.zeros((len(U) + 1, len(x0)))
    x[0] = x0
    for k in range(len(U)):
        x[k + 1] = A @ x[k] + B @ U[k] + d
    return x


def identify(data, train_frac=0.7, affine=True):
    X, U = data["X"], data["U"]
    N = len(U)
    ntr = int(train_frac * N)
    A, B, d = dmdc(X[:ntr].T, X[1:ntr + 1].T, U[:ntr].T, affine=affine)
    # validação em simulação livre (como lsim em dmdc_3.m)
    xtr = simulate_linear(A, B, d, X[0], U[:ntr])
    xval = simulate_linear(A, B, d, X[ntr], U[ntr:])
    rmse = np.sqrt(np.mean((xval - X[ntr:]) ** 2, axis=0))
    fit = 100 * (1 - np.linalg.norm(xval - X[ntr:], axis=0) /
                 np.linalg.norm(X[ntr:] - X[ntr:].mean(0), axis=0))
    return dict(A=A, B=B, d=d, dt=data["dt"], ntr=ntr, xtr=xtr, xval=xval,
                rmse=rmse, fit=fit, eig=np.linalg.eigvals(A))


def plot_identification(data, model, fname="identificacao_dmdc.png"):
    import matplotlib.pyplot as plt
    t = np.arange(len(data["X"])) * data["dt"]
    ntr = model["ntr"]
    fig, axs = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    scale = [1, 1, 1, 1, 180 / np.pi, 180 / np.pi, 180 / np.pi, 180 / np.pi]
    units = ["", "", "", "", " [°]", " [°]", " [°/s]", " [°/s]"]
    for i, ax in enumerate(axs.flat):
        ax.plot(t, data["X"][:, i] * scale[i], "k", lw=1.2, label="MyoSuite")
        ax.plot(t[:ntr + 1], model["xtr"][:, i] * scale[i], "r-.", lw=1, label="DMDc treino")
        ax.plot(t[ntr:], model["xval"][:, i] * scale[i], "g-.", lw=1, label="DMDc validação")
        ax.set_title(f"{STATE_NAMES[i]}{units[i]}  (fit val. {model['fit'][i]:.0f}%)", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("tempo [s]")
    if "ref" in data:
        axs[2, 0].plot(t[:-1], np.rad2deg(data["ref"][:, 0]), "b:", lw=1)
        axs[2, 1].plot(t[:-1], np.rad2deg(data["ref"][:, 1]), "b:", lw=1)
    fig.suptitle("Identificação DMDc da planta MyoSuite (punho)")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    return fig


# =====================================================================
#  3 GDL — experimento de pequena amplitude (banda do tremor) + SINDYc
# =====================================================================
Q_OP = np.deg2rad([0.0, 5.0, 20.0])       # ponto de operação [theta psi phi]


def multisine(t, band, rng, df=0.1, tilt=0.0):
    """
    Multisenoide de fase aleatória em 'band' [Hz], RMS unitário.
    tilt: amplitude de cada componente ∝ f**tilt (pré-ênfase que compensa o
    caráter passa-baixa da planta e concentra energia angular na banda do tremor).
    """
    f = np.arange(band[0], band[1] + 1e-9, df)
    ph = rng.uniform(0, 2 * np.pi, len(f))
    A = (f / f[0]) ** tilt
    s = (A[None, :] * np.sin(2 * np.pi * f[None, :] * t[:, None] + ph[None, :])).sum(1)
    return s / s.std()


def run_tremor_band_experiment(plant, T=60.0, q_op=Q_OP, tau_amp=(0.33, 0.30, 0.365),
                               band=(1.0, 15.0), tilt=0.0, prbs_amp=0.01, taulim=(0.01, 0.05),
                               mus_amp=0.05, ref_amp_deg=(3.0, 3.0, 3.0), ref_period=(1.5, 3.0),
                               base=0.03, gains=(6.0, 0.3, 10.0), settle=2.0, seed=0):
    """
    Experimento de PEQUENA amplitude em torno do ponto de operação (tremor).
    Um controle postural PD+I ('voluntário'), mapeado aos músculos pelos braços
    de momento, sustenta a postura (em malha aberta o punho deriva dezenas de
    graus); somam-se:
      - perturbação de torque multisenoidal por GDL em 'band' (1-15 Hz);
      - multisenoide INDEPENDENTE por músculo (mus_amp, escalada pela
        capacidade de torque de cada músculo) -> descorrelaciona sinérgicos
        (PT/PQ, ECRL/ECRB, ...) e torna cada coluna de B identificável;
      - PRBS de baixa amplitude por músculo;
      - degraus lentos e pequenos da postura de referência (ref_amp_deg, a cada
        ref_period s) -> o equilíbrio estático varia com q e a rigidez
        (gravidade + passiva + fl(l(q))) torna-se identificável.
    Os sinais do controlador são guardados para validar o modelo identificado
    NA MESMA MALHA FECHADA (replay_closed_loop). A fase de assentamento
    (settle s) é descartada.
    """
    rng = np.random.default_rng(seed)
    p = plant
    Ns, N = int(round(settle / p.dt)), int(round(T / p.dt))
    t = np.arange(N) * p.dt
    tau_p = np.stack([a * multisine(t, band, rng, tilt=tilt) for a in tau_amp], 1)
    pr = np.stack([prbs(t, taulim, [-prbs_amp, prbs_amp], rng, T)
                   for _ in range(p.n_mus)], 1)
    x = p.reset(q0=q_op)
    Rw = p.moment_arms() * p.F0            # Nm por unidade de ativação (postura)
    Rn = Rw / np.sum(Rw ** 2, axis=0, keepdims=True).clip(1e-9)
    cap = np.linalg.norm(Rw, axis=0)
    ms = np.stack([multisine(t, band, rng) for _ in range(p.n_mus)], 1)
    pr = pr + mus_amp * (cap.mean() / cap)[None, :].clip(0.5, 3.0) * ms
    # degraus lentos da referência postural
    qref = np.tile(np.asarray(q_op, float), (N, 1))
    if np.any(np.asarray(ref_amp_deg) > 0):
        tk = 0.0
        while tk < T:
            dur = rng.uniform(*ref_period)
            sl = (t >= tk) & (t < tk + dur)
            qref[sl] = q_op + np.deg2rad(rng.uniform(-1, 1, p.n_q) * np.asarray(ref_amp_deg))
            tk += dur
    ctrl = dict(q_op=np.asarray(q_op), qref=qref, Rn=Rn, base=base, gains=np.asarray(gains),
                tau_p=tau_p, pr=pr, iq=p.iq, iv=p.iv, dt=p.dt)
    integ = np.zeros(p.n_q)
    for k in range(Ns):                    # assentamento (sem excitação)
        u, integ = postural_command(ctrl, x, integ, None)
        x = p.step(u)
    X = np.zeros((N + 1, len(p.state_names)))
    U = np.zeros((N, p.n_mus))
    INTEG = np.zeros((N + 1, p.n_q))
    X[0], INTEG[0] = x, integ
    for k in range(N):
        U[k], integ = postural_command(ctrl, X[k], integ, k)
        X[k + 1], INTEG[k + 1] = p.step(U[k]), integ
    return dict(t=t, X=X, U=U, dt=p.dt, q_op=np.asarray(q_op), INTEG=INTEG,
                **{"ctrl_" + k: v for k, v in ctrl.items()})


def postural_command(ctrl, x, integ, k):
    """Controle postural PD+I + excitação (k=None: sem excitação)."""
    kp, kd, ki = ctrl["gains"]
    r = ctrl["q_op"] if k is None or "qref" not in ctrl else ctrl["qref"][k]
    e = r - x[ctrl["iq"]]
    integ = integ + e * ctrl["dt"]
    tau = kp * e - kd * x[ctrl["iv"]] + ki * integ
    if k is not None:
        tau = tau + ctrl["tau_p"][k]
    u = ctrl["base"] + np.maximum(ctrl["Rn"].T @ tau, 0)
    if k is not None:
        u = u + ctrl["pr"][k]
    return np.clip(u, 0, 1), integ


def replay_closed_loop(step_fn, data, k0, k1=None):
    """
    Simulação livre do modelo (step_fn(x,u)->x+) na MESMA malha fechada do
    experimento (mesmo controle postural e mesmas excitações), a partir de k0.
    """
    ctrl = {k[5:]: v for k, v in data.items() if k.startswith("ctrl_")}
    k1 = k1 or len(data["U"])
    x, integ = data["X"][k0].copy(), data["INTEG"][k0].copy()
    Xs = np.full((k1 - k0 + 1, len(x)), np.nan)
    Xs[0] = x
    for i, k in enumerate(range(k0, k1)):
        u, integ = postural_command(ctrl, x, integ, k)
        x = step_fn(x, u)
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e3:
            break
        Xs[i + 1] = x
    return Xs


# ---------------------------------------------------------------- bibliotecas
def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def lib_act(a, u):
    """Biblioteca da dinâmica de ativação (por músculo). a, u: (N, n_mus) -> (N, n_mus, 10)."""
    rp, rn = np.maximum(u - a, 0), np.maximum(a - u, 0)
    one = np.ones_like(a)
    return np.stack([one, a, u, a * a, a * u, u * u, rp, rp * a, rn, rn * a], -1)


LIB_ACT_NAMES = ["1", "a", "u", "a^2", "a*u", "u^2", "(u-a)+", "(u-a)+*a", "(a-u)+", "(a-u)+*a"]


def lib_mech(q, v, a, q_ref=None, trig=True):
    """
    Biblioteca da dinâmica mecânica (comum a q e qdot). q, v: (N,n_q); a: (N,n_mus).
    Torque muscular ~ a*F0*fl(l(q))*fv(v)*r(q) -> termos a, a*q, a*v, a*q*q;
    gravidade -> sin/cos(q); passivo/amortecimento -> q, v, q^2, q*v, v^2.
    q_ref: centra os termos polinomiais no ponto de operação (dq = q - q_ref),
           essencial p/ dados de pequena amplitude (senão a, a*q, a*q^2 ficam colineares).
    """
    N, nq = q.shape
    qa = q
    if q_ref is not None:
        q = q - q_ref
    pq = _pairs(nq)
    qq = np.stack([q[:, i] * q[:, j] for i, j in pq], 1)
    vv = np.stack([v[:, i] * v[:, j] for i, j in pq], 1)
    qv = (q[:, :, None] * v[:, None, :]).reshape(N, -1)
    aq = (a[:, :, None] * q[:, None, :]).reshape(N, -1)
    av = (a[:, :, None] * v[:, None, :]).reshape(N, -1)
    aqq = (a[:, :, None] * qq[:, None, :]).reshape(N, -1)
    trig_terms = [np.sin(qa), np.cos(qa)] if trig else []
    return np.hstack([np.ones((N, 1)), q, v, qq, qv, vv, *trig_terms, a, aq, av, aqq])


def lib_mech_names(coords, muscles, trig=True):
    nq = len(coords)
    c = coords
    ms = [m.lower() for m in muscles]
    pq = _pairs(nq)
    n = ["1"] + c + [x + "dot" for x in c]
    n += [f"{c[i]}*{c[j]}" for i, j in pq]
    n += [f"{c[i]}*{c[j]}dot" for i in range(nq) for j in range(nq)]
    n += [f"{c[i]}dot*{c[j]}dot" for i, j in pq]
    if trig:
        n += [f"sin({x})" for x in c] + [f"cos({x})" for x in c]
    n += [f"a_{m}" for m in ms]
    n += [f"a_{m}*{x}" for m in ms for x in c]
    n += [f"a_{m}*{x}dot" for m in ms for x in c]
    n += [f"a_{m}*{c[i]}*{c[j]}" for m in ms for i, j in pq]
    return n


# ---------------------------------------------------------------- SINDYc
class SINDYcModel:
    """x(k+1) = x(k) + [Theta_act Xi_act ; Theta_mech Xi_mech]  (tempo discreto)."""

    def __init__(self, n_mus, n_q, dt, q_ref=None, trig=False, ridge=1e-6, protect_linear=True):
        self.n_mus, self.n_q, self.dt = n_mus, n_q, dt
        self.q_ref = None if q_ref is None else np.asarray(q_ref, float)
        self.trig, self.ridge, self.protect_linear = trig, ridge, protect_linear

    def _lm(self, q, v, a):
        return lib_mech(q, v, a, self.q_ref, self.trig)

    def fit(self, X, U, lam_act, lam_mech):
        """
        lam_*: escalar ou vetor (um λ por estado).
        X, U: um registro (X (N+1,n), U (N,m)) ou LISTAS de episódios.
        """
        na, nq = self.n_mus, self.n_q
        if isinstance(X, (list, tuple)):
            dX = np.vstack([x[1:] - x[:-1] for x in X])
            Xk = np.vstack([x[:-1] for x in X])
            U = np.vstack(U)
        else:
            dX = X[1:] - X[:-1]
            Xk = X[:-1]
        a, q, v = Xk[:, :na], Xk[:, na:na + nq], Xk[:, na + nq:]
        La = lib_act(a, U)
        lam_act = np.broadcast_to(lam_act, (na,))
        kw = dict(ref="target", ridge=self.ridge)
        self.Xi_act = np.stack([stlsq(La[:, i], dX[:, [i]], lam_act[i], **kw)[:, 0]
                                for i in range(na)], 1)            # (10, na)
        Lm = self._lm(q, v, a)
        lam_mech = np.broadcast_to(lam_mech, (2 * nq,))
        # restrição física: nas equações de ACELERAÇÃO a parte linear
        # [1, q, qdot, a] é protegida da poda (rigidez/gravidade dominam a
        # baixa frequência mas contribuem pouco para dqdot em 1 passo)
        keep = np.zeros(Lm.shape[1], bool)
        keep[:1 + 2 * nq] = True                                      # 1, q, qdot
        i_a = Lm.shape[1] - na * (1 + nq + nq + len(_pairs(nq)))      # início dos termos em a
        keep[i_a:i_a + na] = True                                     # a
        self.keep_mech = keep if self.protect_linear else None
        self.Xi_mech = np.stack([stlsq(Lm, dX[:, [na + j]], lam_mech[j],
                                       keep=self.keep_mech if j >= nq else None, **kw)[:, 0]
                                 for j in range(2 * nq)], 1)        # (p, 2nq)
        return self

    @property
    def nnz(self):
        return int(np.count_nonzero(self.Xi_act) + np.count_nonzero(self.Xi_mech))

    def step(self, x, u):
        """Um passo do modelo; x: (n,), u: (n_mus,)."""
        na, nq = self.n_mus, self.n_q
        a, q, v = x[None, :na], x[None, na:na + nq], x[None, na + nq:]
        da = np.einsum("nk,kn->n", lib_act(a, u[None])[0], self.Xi_act)
        dm = self._lm(q, v, a)[0] @ self.Xi_mech
        xn = x + np.concatenate([da, dm])
        xn[:na] = np.clip(xn[:na], 0, 1)
        return xn

    def simulate(self, x0, U):
        x = np.full((len(U) + 1, len(x0)), np.nan)
        x[0] = x0
        for k in range(len(U)):
            x[k + 1] = self.step(x[k], U[k])
            if not np.all(np.isfinite(x[k + 1])) or np.abs(x[k + 1]).max() > 1e3:
                break
        return x

    def save(self, fname):
        np.savez(fname, Xi_act=self.Xi_act, Xi_mech=self.Xi_mech, n_mus=self.n_mus,
                 n_q=self.n_q, dt=self.dt, fit=getattr(self, "fit", np.nan),
                 q_ref=self.q_ref if self.q_ref is not None else np.nan, trig=self.trig)

    @classmethod
    def load(cls, fname):
        z = np.load(fname)
        qr = z["q_ref"]
        m = cls(int(z["n_mus"]), int(z["n_q"]), float(z["dt"]),
                q_ref=None if np.all(np.isnan(qr)) else qr, trig=bool(z["trig"]))
        m.Xi_act, m.Xi_mech, m.fit = z["Xi_act"], z["Xi_mech"], z["fit"]
        return m


def fit_pct(xs, xr):
    with np.errstate(invalid="ignore"):
        f = 100 * (1 - np.linalg.norm(xs - xr, axis=0) / np.linalg.norm(xr - xr.mean(0), axis=0))
    return np.where(np.isfinite(f), f, -np.inf)


def identify_sindyc(data, n_mus, n_q, fit_min=80.0, train_frac=0.7,
                    lam_grid=np.logspace(-5, -0.5, 19), q_ref=None, trig=False,
                    ridge=1e-6, closed_loop_val=True, protect_linear=True, verbose=True):
    """
    Esparsifica o SINDYc até o limite fit_min (em TODOS os estados, validação
    em simulação livre — por padrão reproduzindo a malha fechada do experimento):
      1) λ global: maior valor do grid com min(fit) >= fit_min
      2) refinamento guloso: aumenta o λ de cada estado enquanto min(fit) >= fit_min
    """
    X, U, dt = data["X"], data["U"], data["dt"]
    ntr = int(train_frac * len(U))
    Xtr, Utr, Xva, Uva = X[:ntr + 1], U[:ntr], X[ntr:], U[ntr:]
    ns = 2 * n_q
    log = []

    def evaluate(la, lm):
        mdl = SINDYcModel(n_mus, n_q, dt, q_ref=q_ref, trig=trig, ridge=ridge,
                          protect_linear=protect_linear).fit(Xtr, Utr, la, lm)
        f = fit_pct(validate(mdl.step), Xva)
        return mdl, f

    def validate(step_fn):
        if closed_loop_val:
            return replay_closed_loop(step_fn, data, ntr)
        x = np.full_like(Xva, np.nan); x[0] = Xva[0]
        for k in range(len(Uva)):
            x[k + 1] = step_fn(x[k], Uva[k])
        return x

    best = None
    for lam in lam_grid:
        mdl, f = evaluate(lam, lam)
        log.append((lam, mdl.nnz, f.min()))
        if verbose:
            print(f"   λ={lam:8.2e}  termos={mdl.nnz:5d}  min fit={f.min():7.1f}%")
        if f.min() >= fit_min:
            best = (lam, mdl, f)
        elif best is not None:
            break
    if best is None:
        raise RuntimeError("nenhum λ atinge o fit mínimo — revise biblioteca/experimento")
    lam0, mdl, f = best
    la, lm = np.full(n_mus, lam0), np.full(ns, lam0)

    grid_up = lam_grid[lam_grid > lam0]
    for group, vec, n in [("act", la, n_mus), ("mech", lm, ns)]:
        for i in range(n):
            for lam in grid_up:
                trial = vec.copy()
                trial[i] = lam
                m2, f2 = evaluate(trial if group == "act" else la,
                                  trial if group == "mech" else lm)
                if f2.min() >= fit_min and m2.nnz < mdl.nnz:
                    vec[i] = lam
                    mdl, f = m2, f2
                else:
                    break
    if verbose:
        print(f"   refinado por estado: termos={mdl.nnz}  min fit={f.min():.1f}%")
    mdl.lam_act, mdl.lam_mech, mdl.fit, mdl.ntr = la, lm, f, ntr
    mdl.xval = validate(mdl.step)
    return mdl, log


def dmdc_step_fn(A, B, d):
    return lambda x, u: A @ x + B @ u + d


# =====================================================================
#  Identificação em MALHA ABERTA (episódica) — palma voltada para o solo
# =====================================================================
Q_PALM_DOWN = np.deg2rad([0.0, 0.0, 80.0])   # [theta psi phi]: mão e antebraço horizontais,
                                             # palma p/ o solo (13° de resíduo; limite φ = 90°)

# tremor: excitação alternada agonista/antagonista (flexores x extensores,
# pronadores x supinador), como o oscilador da tese somava às excitações
TREMOR_GROUPS = {"flex": ["FCR", "FCU"], "ext": ["ECRL", "ECRB", "ECU"],
                 "pron": ["PT", "PQ"], "sup": ["SUP"]}


def tremor_excitation(plant, t, f=5.0, amp=0.08):
    """Excitação de tremor (n_mus,) no instante t: meia-onda em antifase por grupo."""
    s = np.sin(2 * np.pi * f * t)
    u = np.zeros(plant.n_mus)
    for g, sign in [("flex", +1), ("ext", -1), ("pron", +1), ("sup", -1)]:
        for mname in TREMOR_GROUPS[g]:
            u[plant.muscles.index(mname)] = amp * max(sign * s, 0.0)
    return u


def find_operating_point(plant, q_op=Q_PALM_DOWN, base=0.15, gains=(6.0, 0.3, 10.0), T=4.0):
    """
    Leva o punho a q_op com um controle postural PD+I (apenas para ACHAR o
    ponto de operação) e devolve o estado MuJoCo assentado e as excitações
    de equilíbrio u* (co-contração ~base). Depois disso o controlador é desligado.
    """
    p = plant
    x = p.reset(q0=q_op, a0=base)
    Rw = p.moment_arms() * p.F0
    Rn = Rw / np.sum(Rw ** 2, axis=0, keepdims=True).clip(1e-9)
    ctrl = dict(q_op=np.asarray(q_op), Rn=Rn, base=base, gains=np.asarray(gains),
                iq=p.iq, iv=p.iv, dt=p.dt)
    integ = np.zeros(p.n_q)
    for _ in range(int(T / p.dt)):
        u, integ = postural_command(ctrl, x, integ, None)
        x = p.step(u)
    state = (p.d.qpos.copy(), p.d.qvel.copy(), p.d.act.copy())
    return state, u, x


def run_openloop_episodes(plant, n_ep=40, T_ep=1.5, q_op=Q_PALM_DOWN, base=0.15,
                          mus_amp=0.04, band=(1.0, 15.0), prbs_amp=0.01, taulim=(0.01, 0.05),
                          drift_lim_deg=10.0, seed=0):
    """
    Identificação em MALHA ABERTA pura: cada episódio parte do estado de
    equilíbrio (achado uma única vez por find_operating_point) e aplica
        u(t) = u* + multisenoide independente por músculo (1-15 Hz) + PRBS
    SEM realimentação. Como a planta é instável em malha aberta nessa postura
    (modo flexão-desvio, σ ≈ +1,3 1/s), cada episódio é curto e é encerrado se
    |q - q_op| > drift_lim_deg.
    """
    rng = np.random.default_rng(seed)
    p = plant
    (qpos0, qvel0, act0), u_eq, x_eq = find_operating_point(p, q_op, base)
    cap = np.linalg.norm(p.moment_arms() * p.F0, axis=0)
    scale = (cap.mean() / cap).clip(0.5, 5.0)
    N = int(round(T_ep / p.dt))
    t = np.arange(N) * p.dt
    Xs, Us = [], []
    for e in range(n_ep):
        ms = np.stack([multisine(t, band, rng) for _ in range(p.n_mus)], 1)
        pr = np.stack([prbs(t, taulim, [-prbs_amp, prbs_amp], rng, T_ep) for _ in range(p.n_mus)], 1)
        Uep = np.clip(u_eq + mus_amp * scale * ms + pr, 0, 1)
        mujoco.mj_resetData(p.m, p.d)
        p.d.qpos[:], p.d.qvel[:], p.d.act[:] = qpos0, qvel0, act0
        mujoco.mj_forward(p.m, p.d)
        X = [p.state()]
        for k in range(N):
            X.append(p.step(Uep[k]))
            if np.rad2deg(np.abs(X[-1][p.iq] - q_op)).max() > drift_lim_deg:
                break
        X = np.array(X)
        Xs.append(X)
        Us.append(Uep[:len(X) - 1])
    return dict(Xs=Xs, Us=Us, u_eq=u_eq, x_eq=x_eq, q_op=np.asarray(q_op), dt=p.dt,
                state0=(qpos0, qvel0, act0))


def _sparsify(evaluate, n_mus, ns, lam_grid, fit_min, verbose):
    """Laço de esparsificação: λ global (maior com min fit >= fit_min) + refino por estado."""
    best = None
    for lam in lam_grid:
        mdl, f = evaluate(lam, lam)
        if verbose:
            print(f"   λ={lam:8.2e}  termos={mdl.nnz:5d}  min fit={f.min():7.1f}%")
        if f.min() >= fit_min:
            best = (lam, mdl, f)
        elif best is not None:
            break
    if best is None:
        raise RuntimeError("nenhum λ atinge o fit mínimo — revise biblioteca/experimento")
    lam0, mdl, f = best
    la, lm = np.full(n_mus, lam0), np.full(ns, lam0)
    for group, vec, n in [("act", la, n_mus), ("mech", lm, ns)]:
        for i in range(n):
            for lam in lam_grid[lam_grid > lam0]:
                trial = vec.copy()
                trial[i] = lam
                m2, f2 = evaluate(trial if group == "act" else la, trial if group == "mech" else lm)
                if f2.min() >= fit_min and m2.nnz < mdl.nnz:
                    vec[i] = lam
                    mdl, f = m2, f2
                else:
                    break
    if verbose:
        print(f"   refinado por estado: termos={mdl.nnz}  min fit={f.min():.1f}%")
    mdl.lam_act, mdl.lam_mech, mdl.fit = la, lm, f
    return mdl


def simulate_episodes(step_fn, Xs, Us):
    """Simulação livre em MALHA ABERTA de cada episódio (mesmas entradas, mesmo x0)."""
    sims = []
    for X, U in zip(Xs, Us):
        x = np.full_like(X, np.nan)
        x[0] = X[0]
        for k in range(len(U)):
            x[k + 1] = step_fn(x[k], U[k])
            if not np.all(np.isfinite(x[k + 1])) or np.abs(x[k + 1]).max() > 1e3:
                break
        sims.append(x)
    return sims


def identify_sindyc_openloop(exp, n_mus, n_q, fit_min=80.0, train_frac=0.7,
                             lam_grid=np.logspace(-5, -0.5, 19), ridge=1e-6,
                             protect_linear=True, verbose=True):
    """SINDYc com episódios de malha aberta; validação = simulação livre em malha aberta
    dos episódios não usados no treino (fit calculado sobre os episódios concatenados)."""
    Xs, Us, dt = exp["Xs"], exp["Us"], exp["dt"]
    ntr = int(round(train_frac * len(Xs)))
    Xtr, Utr, Xva, Uva = Xs[:ntr], Us[:ntr], Xs[ntr:], Us[ntr:]
    Xva_cat = np.vstack(Xva)

    def evaluate(la, lm):
        mdl = SINDYcModel(n_mus, n_q, dt, q_ref=exp["q_op"], ridge=ridge,
                          protect_linear=protect_linear).fit(Xtr, Utr, la, lm)
        sim = np.vstack(simulate_episodes(mdl.step, Xva, Uva))
        return mdl, fit_pct(sim, Xva_cat)

    mdl = _sparsify(evaluate, n_mus, 2 * n_q, lam_grid, fit_min, verbose)
    mdl.n_train_ep = ntr
    mdl.xval_eps = simulate_episodes(mdl.step, Xva, Uva)
    return mdl


def dmdc_openloop(exp, train_frac=0.7):
    """DMDc afim (referência) com os mesmos episódios e a mesma validação."""
    Xs, Us = exp["Xs"], exp["Us"]
    ntr = int(round(train_frac * len(Xs)))
    X = np.hstack([x[:-1].T for x in Xs[:ntr]])
    Xp = np.hstack([x[1:].T for x in Xs[:ntr]])
    U = np.hstack([u.T for u in Us[:ntr]])
    A, B, d = dmdc(X, Xp, U, affine=True)
    sims = simulate_episodes(lambda x, u: A @ x + B @ u + d, Xs[ntr:], Us[ntr:])
    return dict(A=A, B=B, d=d, fit=fit_pct(np.vstack(sims), np.vstack(Xs[ntr:])), xval_eps=sims)


def plot_openloop_identification(exp, mdl, dm, names, fname="identificacao_malha_aberta.png",
                                 n_show=3):
    import matplotlib.pyplot as plt
    na, nq = mdl.n_mus, mdl.n_q
    ntr = mdl.n_train_ep
    Xva = exp["Xs"][ntr:ntr + n_show]
    sel = list(range(na, na + 2 * nq)) + [names.index("a_ecrl"), names.index("a_fcu"),
                                           names.index("a_pq"), names.index("a_pt")]
    fig, axs = plt.subplots(5, 2, figsize=(13, 12))
    for ax, i in zip(axs.T.flat, sel):
        s = 180 / np.pi if i >= na else 1
        t0 = 0.0
        for e, X in enumerate(Xva):
            t = t0 + np.arange(len(X)) * exp["dt"]
            ax.plot(t, X[:, i] * s, "k", lw=1.2, label="MyoSuite" if e == 0 else None)
            ax.plot(t, dm["xval_eps"][e][:, i] * s, "g-.", lw=0.9,
                    label=f"DMDc ({dm['fit'][i]:.0f}%)" if e == 0 else None)
            ax.plot(t, mdl.xval_eps[e][:, i] * s, "m--", lw=1,
                    label=f"SINDYc ({mdl.fit[i]:.0f}%)" if e == 0 else None)
            ax.axvline(t0, color="0.7", lw=0.8)
            t0 = t[-1] + 0.05
        unit = " [°]" if na <= i < na + nq else (" [°/s]" if i >= na else "")
        ax.set_title(names[i] + unit, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axs[-1]:
        ax.set_xlabel(f"tempo [s] — {n_show} episódios de validação (malha aberta, simulação livre)")
    fig.suptitle(f"Identificação em MALHA ABERTA (palma para baixo) — SINDYc {mdl.nnz} termos, "
                 f"min fit {mdl.fit.min():.0f}%")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)


def plot_sindyc_3dof(data, mdl, dmdc_val, names, fname="identificacao_sindyc_3dof.png"):
    import matplotlib.pyplot as plt
    t = np.arange(len(data["X"])) * data["dt"]
    ntr, nq, na = mdl.ntr, mdl.n_q, mdl.n_mus
    sel = list(range(na, na + 2 * nq)) + [1, 5, 6, 7]     # q, qdot e 4 ativações
    fig, axs = plt.subplots(5, 2, figsize=(13, 12), sharex=True)
    tv = t[ntr:]
    for ax, i in zip(axs.T.flat, sel):
        s = 180 / np.pi if i >= na else 1
        ax.plot(tv, data["X"][ntr:, i] * s, "k", lw=1.2, label="MyoSuite")
        if dmdc_val is not None:
            ax.plot(tv, dmdc_val["xval"][:, i] * s, "g-.", lw=0.9,
                    label=f"DMDc ({dmdc_val['fit'][i]:.0f}%)")
        ax.plot(tv, mdl.xval[:, i] * s, "m--", lw=1, label=f"SINDYc ({mdl.fit[i]:.0f}%)")
        unit = " [°]" if na <= i < na + nq else (" [°/s]" if i >= na else "")
        ax.set_title(names[i] + unit, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(tv[0], tv[0] + 3)
    for ax in axs[-1]:
        ax.set_xlabel("tempo [s] (3 s iniciais da validação)")
    fig.suptitle(f"Identificação 3 GDL em pequena amplitude — SINDYc {mdl.nnz} termos "
                 f"(min fit {mdl.fit.min():.0f}%)")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    data = run_identification_experiment()
    print(f"experimento: {len(data['U'])} amostras em {time.perf_counter()-t0:.1f} s")
    model = identify(data)
    np.set_printoptions(precision=4, suppress=True)
    print("RMSE validação:", dict(zip(STATE_NAMES, np.round(model["rmse"], 4))))
    print("fit validação [%]:", np.round(model["fit"], 1))
    print("|autovalores| de A:", np.round(np.sort(np.abs(model["eig"])), 5))
    np.savez("dmdc_model.npz", **{k: model[k] for k in ["A", "B", "d", "dt", "rmse", "fit"]})
    np.savez("idtf_data.npz", **data)
    plot_identification(data, model)
