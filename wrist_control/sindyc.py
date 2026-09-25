"""
SINDYc (Brunton, Proctor & Kutz 2016) em tempo discreto para a planta do punho.
Porta de 04_DMDc_IDTF/utils/sparsifyDynamics.m (STLSQ) com biblioteca
"informada pela física":
  - polinômios de grau 2 em [x, u]            (acoplamentos a*q, a*qdot, ...)
  - sin/cos de theta e phi                     (gravidade)
  - relu(u-a), relu(a-u), e seus produtos por a
                                             (ativação assimétrica: tau depende de u>a e de a)
Modelo:  x(k+1) = x(k) + Theta(x(k), u(k)) Xi
"""
import itertools

import numpy as np

from plant import STATE_NAMES


def library(X, U):
    """X: (N,8), U: (N,4) na ordem [ECRL FCU PQ SUP]; retorna Theta (N,p) e nomes."""
    a = X[:, :4]                          # [asup aecrl afcu apq]
    ua = U[:, [3, 0, 1, 2]]               # u reordenado p/ casar com a: [SUP ECRL FCU PQ]
    Z = np.hstack([X, U])
    zn = STATE_NAMES + ["u_ecrl", "u_fcu", "u_pq", "u_sup"]
    cols, names = [np.ones(len(X))], ["1"]
    for i in range(Z.shape[1]):
        cols.append(Z[:, i]); names.append(zn[i])
    for i, j in itertools.combinations_with_replacement(range(Z.shape[1]), 2):
        cols.append(Z[:, i] * Z[:, j]); names.append(f"{zn[i]}*{zn[j]}")
    for i, n in [(4, "theta"), (5, "phi")]:
        cols += [np.sin(X[:, i]), np.cos(X[:, i])]; names += [f"sin({n})", f"cos({n})"]
    up, dn = np.maximum(ua - a, 0), np.maximum(a - ua, 0)
    for i in range(4):
        m = STATE_NAMES[i][2:]
        cols += [up[:, i], up[:, i] * a[:, i], dn[:, i], dn[:, i] * a[:, i]]
        names += [f"relu(u-a)_{m}", f"relu(u-a)*a_{m}", f"relu(a-u)_{m}", f"relu(a-u)*a_{m}"]
    return np.column_stack(cols), names


def stlsq(Theta, Y, lam, n_iter=10, ref="max", ridge=0.0, keep=None):
    """
    Sequential thresholded least squares (STLSQ/STRidge) com colunas normalizadas.
    ref="max"   : zera |xi| < lam * max|xi|              (usado no 2 GDL)
    ref="target": zera |xi| < lam * ||y||  -> termo cuja contribuição é menor
                  que uma fração lam do sinal-alvo (robusto à colinearidade)
    ridge       : regularização de Tikhonov relativa (STRidge)
    keep        : máscara (n_termos,) de termos protegidos (nunca podados)
    """
    s = np.linalg.norm(Theta, axis=0); s[s == 0] = 1
    Th = Theta / s

    def solve(M, y):
        if ridge <= 0:
            return np.linalg.lstsq(M, y, rcond=None)[0]
        G = M.T @ M
        return np.linalg.solve(G + ridge * np.trace(G) / len(G) * np.eye(len(G)), M.T @ y)

    Xi = np.column_stack([solve(Th, Y[:, k]) for k in range(Y.shape[1])])
    yn = np.linalg.norm(Y, axis=0)
    for _ in range(n_iter):
        thr = lam * (np.abs(Xi).max(0) if ref == "max" else yn)
        small = np.abs(Xi) < thr
        if keep is not None:
            small[np.asarray(keep, bool)] = False
        Xi[small] = 0
        for k in range(Y.shape[1]):
            big = ~small[:, k]
            if big.any():
                Xi[big, k] = solve(Th[:, big], Y[:, k])
    return Xi / s[:, None]


def simulate(Xi, x0, U):
    x = np.zeros((len(U) + 1, len(x0))); x[0] = x0
    for k in range(len(U)):
        th, _ = library(x[k:k + 1], U[k:k + 1])
        x[k + 1] = x[k] + (th @ Xi)[0]
        x[k + 1, :4] = np.clip(x[k + 1, :4], 0, 1)
        if not np.all(np.isfinite(x[k + 1])) or np.abs(x[k + 1]).max() > 1e3:
            x[k + 1:] = np.nan
            break
    return x


def fit_pct(xs, xr):
    return 100 * (1 - np.linalg.norm(xs - xr, axis=0) / np.linalg.norm(xr - xr.mean(0), axis=0))


def identify(data, lam=1e-3, train_frac=0.7):
    X, U = data["X"], data["U"]
    ntr = int(train_frac * len(U))
    Th, names = library(X[:ntr], U[:ntr])
    Xi = stlsq(Th, X[1:ntr + 1] - X[:ntr], lam)
    xval = simulate(Xi, X[ntr], U[ntr:])
    return dict(Xi=Xi, names=names, ntr=ntr, xval=xval, fit=fit_pct(xval, X[ntr:]),
                nnz=int(np.count_nonzero(Xi)))


if __name__ == "__main__":
    import sysid
    data = dict(np.load("idtf_data.npz"))
    d = sysid.identify(data)
    print("DMDc  fit val [%]:", np.round(d["fit"], 1), " média", np.round(d["fit"].mean(), 1))
    best = None
    for lam in [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]:
        s = identify(data, lam)
        f = s["fit"]
        print(f"SINDYc λ={lam:g}: termos={s['nnz']:4d}  fit val [%]:", np.round(f, 1),
              " média", np.round(np.nanmean(f), 1) if np.all(np.isfinite(f)) else "divergiu")
        if np.all(np.isfinite(f)) and (best is None or f.mean() > best["fit"].mean()):
            best, best["lam"] = s, lam
    np.savez("sindyc_model.npz", Xi=best["Xi"], names=best["names"], lam=best["lam"])

    import matplotlib.pyplot as plt
    t = np.arange(len(data["X"])) * data["dt"]; ntr = best["ntr"]
    fig, axs = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    sc = [1] * 4 + [180 / np.pi] * 4
    for i, ax in enumerate(axs.flat):
        ax.plot(t[ntr:], data["X"][ntr:, i] * sc[i], "k", lw=1.3, label="MyoSuite")
        ax.plot(t[ntr:], d["xval"][:, i] * sc[i], "g-.", lw=1, label=f"DMDc ({d['fit'][i]:.0f}%)")
        ax.plot(t[ntr:], best["xval"][:, i] * sc[i], "m--", lw=1, label=f"SINDYc ({best['fit'][i]:.0f}%)")
        ax.set_title(STATE_NAMES[i], fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"Validação (simulação livre): DMDc x SINDYc (λ={best['lam']:g}, {best['nnz']} termos)")
    fig.tight_layout(); fig.savefig("dmdc_vs_sindyc.png", dpi=130)
