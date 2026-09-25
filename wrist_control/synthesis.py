"""
Síntese do controlador estabilizante H-infinito por sensibilidade mista
sobre o modelo DMDc identificado na planta MyoSuite.
Porta para Python o ControllerSynthesis.m:

    sys = d2c(sysDMDc)
    W1  = makeweight(10,  [30, 1 ], .01)   # pondera S  (baixa frequência)
    W3  = makeweight(.01, [30, .9], 1  )   # pondera T  (alta frequência)
    K   = mixsyn(sys, W1, W2, W3);   Kz = c2d(K, Ts)

Diferença: slycot (SB10AD) exige D12 de posto completo, então usa-se um
W2 constante pequeno (regularização do esforço de controle), em vez de W2=[].

Também calcula o ponto de equilíbrio (x*, u*) do modelo afim para o setpoint,
usado como referência dos estados de ativação e como pré-alimentação.
"""
import control as ct
import numpy as np
from scipy.linalg import logm, schur, solve_sylvester

from plant import STATE_NAMES


def d2c_zoh(A, B, dt):
    """Inversa exata do ZOH (equivalente ao d2c do MATLAB)."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:], M[n:, n:] = A, B, np.eye(m)
    L = np.real(logm(M)) / dt
    return L[:n, :n], L[:n, n:]


def makeweight(dcgain, freq_mag, hfgain):
    """makeweight(dcgain,[freq,mag],hfgain) de 1a ordem (MATLAB)."""
    w, mag = freq_mag
    w0 = w * np.sqrt((hfgain ** 2 - mag ** 2) / (mag ** 2 - dcgain ** 2))
    return ct.tf([hfgain, w0 * dcgain], [1, w0])


def diag_weight(w, n):
    return ct.ss(ct.append(*[ct.ss(w)] * n))


def residualize_fast_modes(K, wmax):
    """
    Remove (por perturbação singular, preservando o ganho DC) os modos do
    controlador com |polo| > wmax — o mixsyn quase-singular gera polos
    de ~1e9 rad/s que não têm significado físico a Ts = 1 ms.
    """
    A, B, C, D = K.A, K.B, K.C, K.D
    T, Z, k = schur(A, output="real", sort=lambda re, im: np.hypot(re, im) <= wmax)
    if k == A.shape[0]:
        return K, 0
    Bt, Ct = Z.T @ B, C @ Z
    As, A12, Af = T[:k, :k], T[:k, k:], T[k:, k:]
    X = solve_sylvester(As, -Af, -A12)            # desacopla os blocos
    Tm = np.block([[np.eye(k), X], [np.zeros((A.shape[0] - k, k)), np.eye(A.shape[0] - k)]])
    Ti = np.linalg.inv(Tm)
    Bt, Ct = Ti @ Bt, Ct @ Tm
    Bs, Bf, Cs, Cf = Bt[:k], Bt[k:], Ct[:, :k], Ct[:, k:]
    Dr = D - Cf @ np.linalg.solve(Af, Bf)
    return ct.ss(As, Bs, Cs, Dr), A.shape[0] - k


def equilibrium(A, B, d, theta_ref, phi_ref, u_base=0.05, reg=1e-3):
    """
    Ponto de equilíbrio do modelo afim  x = A x + B u + d  com
    theta=theta_ref, phi=phi_ref, velocidades nulas; entre as soluções, a de
    menor ||u - u_base|| (co-contração mínima).
    """
    n, m = B.shape
    # incógnitas z = [x (n), u (m)]
    Aeq = np.hstack([np.eye(n) - A, -B])
    Afix = np.zeros((4, n + m))
    Afix[[0, 1, 2, 3], [4, 5, 6, 7]] = 1
    bfix = np.array([theta_ref, phi_ref, 0, 0])
    Mc = np.vstack([Aeq, Afix])
    bc = np.concatenate([d, bfix])
    # mínimos quadrados ponderados: restrições (peso alto) + regularização em u
    R = np.hstack([np.zeros((m, n)), np.eye(m)])
    Mall = np.vstack([Mc / reg, R])
    ball = np.concatenate([bc / reg, np.full(m, u_base)])
    z = np.linalg.lstsq(Mall, ball, rcond=None)[0]
    x_eq, u_eq = z[:n], z[n:]
    return x_eq, np.clip(u_eq, 0, 1)


# ------------------------------------------------------------ SINDYc (N GDL)
def equilibrium_nl(step_fn, n_mus, n_q, q_ref, u_base=0.05, reg=0.05, x_guess=None):
    """
    Ponto de equilíbrio de um modelo não linear discreto x+ = f(x,u) com
    q = q_ref, qdot = 0. Incógnitas: ativações a (n_mus) e excitações u (n_mus);
    entre as soluções (redundância muscular), a de menor ||u - u_base||.
    """
    from scipy.optimize import least_squares
    q_ref = np.asarray(q_ref, float)

    def xfull(a):
        return np.concatenate([a, q_ref, np.zeros(n_q)])

    def res(z):
        a, u = z[:n_mus], z[n_mus:]
        x = xfull(a)
        r = step_fn(x, u) - x
        return np.concatenate([r * 1e3, reg * (u - u_base)])

    z0 = np.full(2 * n_mus, u_base) if x_guess is None else x_guess
    sol = least_squares(res, z0, bounds=(0, 1), xtol=1e-12, ftol=1e-12)
    a, u = sol.x[:n_mus], sol.x[n_mus:]
    resid = np.abs(step_fn(xfull(a), u) - xfull(a)).max()
    return xfull(a), u, resid


def linearize(step_fn, x0, u0, eps=1e-6):
    """Jacobianas A = df/dx, B = df/du por diferenças centrais."""
    n, m = len(x0), len(u0)
    A, B = np.zeros((n, n)), np.zeros((n, m))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (step_fn(x0 + dx, u0) - step_fn(x0 - dx, u0)) / (2 * eps)
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        B[:, j] = (step_fn(x0, u0 + du) - step_fn(x0, u0 - du)) / (2 * eps)
    return A, B


def synthesize(A, B, dt, W1=(10, (30, 1.0), 0.01), W3=(0.01, (30, 0.9), 1.0),
               w2=1e-2, Ts=None, perf_idx=(4, 5), jw_shift=1e-2):
    """
    Retorna controlador discreto (Ak,Bk,Ck,Dk) tal que  du = K(z) e,  e = r - x.
    perf_idx: índices dos estados usados como saídas medidas/performance.
              (4, 5) = (theta, phi) [padrão]; None = os 8 estados, como na tese
              (formulação infactível em baixa freq.: 8 saídas p/ 4 entradas).
    jw_shift: se a planta tem polos sobre o eixo jw (p.ex. integradores q = ∫qdot
              do modelo SINDYc), a síntese é feita sobre G(s - eps) (polos
              deslocados de -eps) — hipótese exigida pelo hinfsyn. A estabilidade
              e a norma são verificadas com a planta ORIGINAL.
    """
    Ts = Ts or dt
    Ac, Bc = d2c_zoh(A, B, dt)
    n, m = Bc.shape
    idx = np.arange(n) if perf_idx is None else np.asarray(perf_idx)
    C = np.eye(n)[idx]
    G = ct.ss(Ac, Bc, C, np.zeros((len(idx), m)))
    ny = len(idx)
    w1 = diag_weight(makeweight(*W1), ny)
    w3 = diag_weight(makeweight(*W3), ny)
    w2s = ct.ss([], [], [], w2 * np.eye(m))
    shifted = np.min(np.abs(np.real(np.linalg.eigvals(Ac)))) < jw_shift
    Gs = ct.ss(Ac - jw_shift * np.eye(n), Bc, C, G.D) if shifted else G
    K, CL, (gamma_rep, rcond) = ct.mixsyn(Gs, w1, w2s, w3)
    K, n_fast = residualize_fast_modes(K, wmax=2.0 / Ts)

    # convenção de sinal: queremos u = +K e com e = r - y
    L = G * K
    cl = ct.feedback(L, np.eye(ny))
    if np.max(np.real(cl.poles())) >= 0:
        K = -K
        cl = ct.feedback(G * K, np.eye(ny))
    stable_ct = np.max(np.real(cl.poles())) < 0

    Kz = ct.c2d(K, Ts, method="zoh")
    # estabilidade da malha discreta com o próprio modelo DMDc
    Gz = ct.ss(A, B, C, np.zeros((ny, m)), dt)
    clz = ct.feedback(Gz * Kz, np.eye(ny))
    stable_dt = np.max(np.abs(clz.poles())) < 1

    # norma H-inf verificada numericamente (o gamma do slycot pode não
    # corresponder à malha real quando o problema é mal posto)
    S = ct.feedback(ct.ss([], [], [], np.eye(ny)), G * K)
    T = ct.feedback(G * K, np.eye(ny))
    wgrid = np.logspace(-2, np.log10(np.pi / Ts), 400)
    hn = lambda sys: max(np.linalg.svd(sys(1j * x), compute_uv=False)[0] for x in wgrid)
    gamma = max(hn(w1 * S), hn(w2s * K * S), hn(w3 * T))

    return dict(Ak=Kz.A, Bk=Kz.B, Ck=Kz.C, Dk=Kz.D, Ts=Ts, gamma=gamma,
                gamma_reported=gamma_rep, n_fast_removed=n_fast,
                perf_idx=idx, stable_ct=stable_ct, stable_dt=stable_dt,
                K=K, G=G, cl_poles_max=np.max(np.abs(clz.poles())),
                ctrl_poles=K.poles())


def plot_loop_shapes(res, W1, W3, fname="sintese_hinf.png"):
    import matplotlib.pyplot as plt
    G, K = res["G"], res["K"]
    ny = G.noutputs
    w = np.logspace(-1, 3.5, 400)
    L = G * K
    S = ct.feedback(ct.ss([], [], [], np.eye(ny)), L)
    T = ct.feedback(L, np.eye(ny))
    sv = lambda sys: np.array([np.linalg.svd(sys(1j * wi), compute_uv=False) for wi in w])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, sys, W, nome in [(axs[0], S, W1, "S"), (axs[1], T, W3, "T")]:
        s = sv(sys)
        ax.semilogx(w, 20 * np.log10(s[:, 0]), "b", label=f"σmax({nome})")
        ax.semilogx(w, 20 * np.log10(s[:, -1]), "b--", lw=0.8, label=f"σmin({nome})")
        wt = makeweight(*W)
        ax.semilogx(w, -20 * np.log10(np.abs(wt(1j * w))), "r", label=f"1/|W{'1' if nome=='S' else '3'}|")
        ax.set_xlabel("ω [rad/s]"); ax.set_ylabel("dB"); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8); ax.set_title(f"{nome}(jω)")
    fig.suptitle(f"Síntese H∞ mixsyn — γ = {res['gamma']:.3f}")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)


if __name__ == "__main__":
    mdl = np.load("dmdc_model.npz")
    A, B, d, dt = mdl["A"], mdl["B"], mdl["d"], float(mdl["dt"])
    W1 = (10, (30, 1.0), 0.01)
    W3 = (0.01, (30, 0.9), 1.0)
    res = synthesize(A, B, dt, W1, W3)
    print(f"gamma (verificado) = {res['gamma']:.4f}   [slycot: {res['gamma_reported']:.4f}]")
    print(f"modos rápidos residualizados: {res['n_fast_removed']}")
    print(f"ordem do controlador = {res['Ak'].shape[0]}")
    print(f"malha fechada estável (contínuo / DMDc discreto): {res['stable_ct']} / {res['stable_dt']}"
          f"   max|polo z| = {res['cl_poles_max']:.5f}")
    print("polo mais rápido do controlador [rad/s]:", np.max(np.abs(res['ctrl_poles'])).round(1))
    x_eq, u_eq = equilibrium(A, B, d, 0.0, np.deg2rad(20))
    print("equilíbrio p/ theta=0°, phi=20°:  u* [ECRL FCU PQ SUP] =", np.round(u_eq, 3))
    print("                                   x* =", dict(zip(STATE_NAMES, np.round(x_eq, 3))))
    np.savez("controller_hinf.npz", Ak=res["Ak"], Bk=res["Bk"], Ck=res["Ck"], Dk=res["Dk"],
             Ts=res["Ts"], gamma=res["gamma"], perf_idx=res["perf_idx"])
    plot_loop_shapes(res, W1, W3)
