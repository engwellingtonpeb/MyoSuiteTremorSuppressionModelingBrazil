"""
Visualizador da planta do punho (myoArm reduzido do MyoSuite).

  python visualize.py                 -> PNG com a condição inicial (ponto de operação)
                                         em 3 vistas, músculos coloridos pela ativação
  python visualize.py --gif           -> também anima a malha fechada (tremor 5 Hz) em GIF
  python visualize.py --interactive   -> abre o visualizador interativo do MuJoCo
                                         (mouse: girar/zoom; duplo-clique: selecionar corpo)

Cores dos tendões: azul (ativação 0) -> vermelho (ativação 1), como no MyoSuite.
"""
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from plant import WristPlantND
from sysid import Q_PALM_DOWN


def color_muscles(plant, act=None):
    """Colore cada tendão muscular pela ativação (azul -> vermelho)."""
    m, d = plant.m, plant.d
    act = d.act[plant.act_idx] if act is None else act
    for i, a_id in enumerate(plant.act_idx):
        t_id = m.actuator_trnid[a_id, 0]
        a = float(np.clip(act[i], 0, 1))
        m.tendon_rgba[t_id] = [0.2 + 0.8 * a, 0.25 * (1 - a), 0.9 * (1 - a), 1.0]
        m.tendon_width[t_id] = 0.002 + 0.003 * a


def set_state(plant, q, act):
    mujoco.mj_resetData(plant.m, plant.d)
    plant.d.qpos[plant.qadr] = q
    plant.d.act[plant.act_idx] = act
    mujoco.mj_forward(plant.m, plant.d)
    color_muscles(plant, act)


def make_camera(plant, azimuth, elevation, distance=0.55, target=("radius", "lunate")):
    """Câmera livre centrada entre o antebraço e a mão."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.mean([plant.d.xpos[plant.m.body(b).id] for b in target], axis=0)
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    return cam


VIEWS = [("lateral (plano sagital)", 180, -10), ("frontal", 90, -10), ("superior", 135, -65)]


def render_views(plant, renderer, views=VIEWS, distance=0.55):
    opt = mujoco.MjvOption()
    opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    opt.sitegroup[:] = 0            # oculta sites (wrapping dos tendões, alvos dos dedos)
    imgs = []
    for _, az, el in views:
        renderer.update_scene(plant.d, camera=make_camera(plant, az, el, distance), scene_option=opt)
        imgs.append(renderer.render().copy())
    return imgs


def snapshot(plant, q, act, fname, title):
    set_state(plant, q, act)
    with mujoco.Renderer(plant.m, height=480, width=480) as r:
        imgs = render_views(plant, r)
    fig, axs = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5.6))
    for ax, im, (nome, _, _) in zip(axs, imgs, VIEWS):
        ax.imshow(im); ax.set_title(nome); ax.axis("off")
    qd = np.rad2deg(q)
    txt = "   ".join(f"{c} = {v:+.1f}°" for c, v in zip(["θ flexão", "ψ desvio", "φ pro-sup"], qd))
    txt2 = "   ".join(f"{n}: {a:.2f}" for n, a in zip(plant.muscles, act))
    fig.suptitle(f"{title}\n{txt}\nativações — {txt2}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print("salvo:", fname)


def initial_condition():
    """Ponto de operação (palma para baixo) e ativações de equilíbrio do controlador salvo."""
    try:
        c = np.load("controller_hinf_3dof.npz")
        return c["q_op"], c["x_eq"][:8], "equilíbrio do modelo SINDYc"
    except FileNotFoundError:
        return Q_PALM_DOWN, np.full(8, 0.15), "co-contração uniforme 0,15"


def closed_loop_gif(plant, fname="tremor_5Hz_planta.gif", T=2.0, fps=25, t_on=0.5):
    """Anima a planta com o controlador H-inf mantendo a postura e o tremor de 5 Hz ligado em t_on."""
    import closed_loop
    import pipeline_3dof_openloop as main_3dof
    import sysid
    from matplotlib.animation import FuncAnimation, PillowWriter
    c = dict(np.load("controller_hinf_3dof.npz"))
    C = main_3dof.make_controller(plant, c)

    def trem(k, x):
        tk = k * plant.dt
        return (sysid.tremor_excitation(plant, tk, main_3dof.F_TREMOR, main_3dof.A_TREMOR)
                if tk >= t_on else np.zeros(plant.n_mus))

    res = closed_loop.run_nd(plant, C, T, lambda t: c["q_op"], q0=c["q_op"], disturbance=trem)
    stride = int(round(1 / (fps * plant.dt)))
    frames = range(0, len(res["U"]), stride)
    renderer = mujoco.Renderer(plant.m, height=360, width=360)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.3))
    ims = [ax.imshow(np.zeros((360, 360, 3), np.uint8)) for ax in axs]
    for ax, (n, _, _) in zip(axs, VIEWS[:2]):
        ax.axis("off"); ax.set_title(n)
    ttl = fig.suptitle("")

    def upd(k):
        set_state(plant, res["X"][k, plant.iq], res["X"][k, :8])
        for im, img in zip(ims, render_views(plant, renderer, VIEWS[:2])):
            im.set_data(img)
        q = np.rad2deg(res["X"][k, plant.iq])
        estado = "tremor ON" if k * plant.dt >= t_on else "tremor OFF"
        ttl.set_text(f"t = {k * plant.dt:.2f} s ({estado})  θ={q[0]:+.1f}°  ψ={q[1]:+.1f}°  φ={q[2]:+.1f}°")
        return ims

    FuncAnimation(fig, upd, frames=frames, blit=False).save(fname, writer=PillowWriter(fps=fps))
    renderer.close()
    print("salvo:", fname)


def interactive(plant, q, act):
    import mujoco.viewer
    set_state(plant, q, act)
    with mujoco.viewer.launch_passive(plant.m, plant.d) as v:
        v.cam.lookat[:] = plant.d.xpos[plant.m.body("lunate").id]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.55, 135, -20
        v.opt.sitegroup[:] = 0
        while v.is_running():
            v.sync()


if __name__ == "__main__":
    plant = WristPlantND("3dof")
    q0, a0, origem = initial_condition()
    snapshot(plant, np.zeros(3), np.full(8, 0.0), "vista_neutra.png",
             "Postura neutra (θ = ψ = φ = 0) — músculos relaxados")
    snapshot(plant, q0, a0, "vista_condicao_inicial.png",
             f"Condição inicial da simulação — ponto de operação ({origem})")
    if "--gif" in sys.argv:
        closed_loop_gif(plant)
    if "--interactive" in sys.argv:
        interactive(plant, q0, a0)
