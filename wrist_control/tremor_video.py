"""
Vídeo do tremor básico: perturbação senoidal de 5 Hz (excitação alternada
agonista/antagonista) aplicada à planta MyoSuite enquanto o controlador H-inf
sustenta a postura de palma para baixo (θ = ψ = 0°, φ = 80°).

Saídas (pasta corrente):
  tremor_5Hz_10s_sagital.mp4, _frontal.mp4, _transversal.mp4   vistas dos 3 planos anatômicos
  tremor_5Hz_10s_3planos.mp4                                     as 3 vistas + traços de θ, ψ, φ
  tremor_5Hz_10s_3planos.gif                                     versão reduzida p/ o README
  tremor_5Hz_10s.npz                                             dados da simulação

Requer controller_hinf_3dof.npz (gerado pelo script 05).
"""
import time

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import closed_loop
import pipeline_3dof_openloop as p3
import sysid
from plant import WristPlantND
from visualize import set_state

T_SIM, T_ON = 10.0, 1.0            # duração [s], início do tremor [s]
FPS = 30
W = H = 480
# planos anatômicos: (nome, azimute, elevação, distância)
PLANES = [("sagital", 180.0, 0.0, 0.34),
          ("frontal", 90.0, 0.0, 0.34),
          ("transversal", 90.0, -89.0, 0.34)]


def simulate(plant, ctrl):
    def trem(k, x):
        tk = k * plant.dt
        return (sysid.tremor_excitation(plant, tk, p3.F_TREMOR, p3.A_TREMOR)
                if tk >= T_ON else np.zeros(plant.n_mus))
    return closed_loop.run_nd(plant, p3.make_controller(plant, ctrl), T_SIM,
                              lambda t: ctrl["q_op"], q0=ctrl["q_op"], disturbance=trem)


def _camera(plant, az, el, dist):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = hand_center(plant)
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    return cam


def hand_center(plant):
    """Centróide das malhas visíveis da mão (corpos a partir do 'lunate')."""
    m, d = plant.m, plant.d
    root = m.body("lunate").id
    in_hand = np.zeros(m.nbody, bool)
    for b in range(m.nbody):
        k = b
        while k > 0 and k != root:
            k = m.body_parentid[k]
        in_hand[b] = k == root
    g = np.nonzero(in_hand[m.geom_bodyid] & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH))[0]
    return d.geom_xpos[g].mean(axis=0)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow antigo
        return ImageFont.load_default()


def _label(img, lines):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    y = 8
    for txt, size in lines:
        dr.text((10, y), txt, fill=(20, 20, 20), font=_font(size),
                stroke_width=2, stroke_fill=(255, 255, 255))
        y += size + 6
    return np.asarray(im)


class TracePanel:
    """Painel com θ, ψ, φ (desvio do ponto de operação) e cursor de tempo."""

    def __init__(self, t, q_dev, width_px, height_px=304):
        matplotlib.use("Agg")
        dpi = 100
        self.fig, self.ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        for j, (lab, c) in enumerate(zip(["θ flexão", "ψ desvio rad./uln.", "φ pro-sup"],
                                         ["tab:blue", "tab:orange", "tab:green"])):
            self.ax.plot(t, q_dev[:, j], color=c, lw=1.0, label=lab)
        self.ax.axvline(T_ON, color="r", ls=":", lw=1)
        self.cur = self.ax.axvline(0, color="k", lw=1.2)
        self.ax.set_xlim(t[0], t[-1])
        self.ax.set_xlabel("tempo [s]")
        self.ax.set_ylabel("ângulo − ponto de operação [°]")
        self.ax.grid(alpha=0.3)
        self.ax.legend(loc="upper left", ncol=3, fontsize=9)
        self.fig.tight_layout()

    def frame(self, tk):
        self.cur.set_xdata([tk, tk])
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def main():
    tic = time.perf_counter()
    plant = WristPlantND("3dof")
    ctrl = dict(np.load("controller_hinf_3dof.npz"))
    print(f"simulando {T_SIM:.0f} s (tremor de {p3.F_TREMOR:.0f} Hz a partir de t = {T_ON:.0f} s) ...")
    res = simulate(plant, ctrl)
    X, t = res["X"], res["t"]
    q_deg = np.rad2deg(X[:, plant.iq])
    q_dev = q_deg - np.rad2deg(ctrl["q_op"])
    k_on = t >= T_ON + 1.0
    print(f"   {res['wall']:.1f} s de simulação; tremor pico-a-pico [θ ψ φ] = "
          f"{(q_deg[k_on].max(0) - q_deg[k_on].min(0)).round(2)}°")
    np.savez("tremor_5Hz_10s.npz", t=t, X=X, U=res["U"], q_op=ctrl["q_op"],
             f_tremor=p3.F_TREMOR, a_tremor=p3.A_TREMOR, t_on=T_ON)

    stride = int(round(1.0 / (FPS * plant.dt)))
    ks = np.arange(0, len(t), stride)
    renderer = mujoco.Renderer(plant.m, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    panel = TracePanel(t, q_dev, width_px=W * len(PLANES))
    kw = dict(fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    writers = {n: imageio.get_writer(f"tremor_5Hz_10s_{n}.mp4", **kw) for n, *_ in PLANES}
    wcomp = imageio.get_writer("tremor_5Hz_10s_3planos.mp4", **kw)
    gif_frames = []
    print(f"renderizando {len(ks)} quadros × {len(PLANES)} vistas ...")
    for i, k in enumerate(ks):
        tk = t[k]
        set_state(plant, X[k, plant.iq], X[k, :plant.n_mus])
        estado = "tremor 5 Hz: LIGADO" if tk >= T_ON else "tremor: desligado"
        views = []
        for nome, az, el, dist in PLANES:
            renderer.update_scene(plant.d, camera=_camera(plant, az, el, dist), scene_option=opt)
            img = _label(renderer.render().copy(),
                         [(f"plano {nome}", 22), (f"t = {tk:5.2f} s   {estado}", 16)])
            writers[nome].append_data(img)
            views.append(img)
        comp = np.vstack([np.hstack(views), panel.frame(tk)])
        wcomp.append_data(comp)
        if i % 2 == 0:                                  # GIF a 15 fps, 640 px (~6 MB)
            gw = 640
            gh = int(round(comp.shape[0] * gw / comp.shape[1]))
            gif_frames.append(Image.fromarray(comp).resize((gw, gh), Image.LANCZOS))
    for w in list(writers.values()) + [wcomp]:
        w.close()
    renderer.close()
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in gif_frames]
    pal[0].save("tremor_5Hz_10s_3planos.gif", save_all=True, append_images=pal[1:],
                duration=int(1000 / (FPS / 2)), loop=0, optimize=True)
    print(f"pronto em {time.perf_counter() - tic:.0f} s")


if __name__ == "__main__":
    main()
