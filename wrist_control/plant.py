"""
Planta biomecânica do punho construída a partir do
myoArm do MyoSuite, reproduzindo o modelo OpenSim usado na tese
(MOBL_ARMS_module2_4_allmuscles_ignoreactivation.osim).

Notação dos graus de liberdade da mão:
    θ (theta) = flexão/extensão       [junta 'flexion']
    φ (phi)   = pronação/supinação    [junta 'pro_sup']
    ψ (psi)   = desvio radial/ulnar   [junta 'deviation']
A ordem interna dos estados segue a ordem das juntas: q = [θ, ψ, φ] (3 GDL) e
q = [θ, φ] (2 GDL). Sinais: +θ = flexão, +ψ = desvio radial, +φ = pronação.

Configurações:
- "2dof": flexion (theta), pro_sup (phi); 7 músculos (tese)      -> WristPlant
- "3dof": flexion (theta), deviation (psi), pro_sup (phi);
          8 músculos (7 da tese + PT)                            -> WristPlantND("3dof")
- Demais juntas travadas na postura do modelo OpenSim e "embutidas" na
  geometria (juntas removidas), o que deixa a simulação bem mais leve.
- Dinâmica de ativação: nativa do MuJoCo, que tem a mesma forma da
  FirstOrderActivationDynamics.m (tau*(0.5+1.5a)); ajustada p/ 12/40 ms.
"""
import os

import mujoco
import numpy as np

import myosuite

MYOARM_XML = os.path.join(os.path.dirname(myosuite.__file__),
                          "simhive", "myo_sim", "arm", "myoarm.xml")

MUSCLES = ["SUP", "ECRL", "ECRB", "ECU", "FCR", "FCU", "PQ"]
FREE_JOINTS = ["flexion", "pro_sup"]          # theta, phi

# postura (default_value das coordenadas travadas no .osim)
POSTURE = {"elv_angle": 1.0472, "shoulder_elv": 0.3491, "shoulder_rot": 0.0,
           "elbow_flexion": 1.2217, "deviation": 0.0}

# músculos comandados pelo controlador (ordem da tese: u1..u4)
CTRL_MUSCLES = ["ECRL", "FCU", "PQ", "SUP"]
# ordem dos estados da tese: x = [asup aecrl afcu apq theta phi thetadot phidot]
STATE_MUSCLES = ["SUP", "ECRL", "FCU", "PQ"]
STATE_NAMES = ["a_sup", "a_ecrl", "a_fcu", "a_pq", "theta", "phi", "thetadot", "phidot"]

CONFIGS = {
    "2dof": dict(free_joints=["flexion", "pro_sup"], coord_names=["theta", "phi"],
                 muscles=MUSCLES),
    "3dof": dict(free_joints=["flexion", "deviation", "pro_sup"],
                 coord_names=["theta", "psi", "phi"],
                 muscles=["SUP", "ECRL", "ECRB", "ECU", "FCR", "FCU", "PQ", "PT"]),
}


def _posture_qpos(m):
    """qpos completo da postura, respeitando os acoplamentos (equality) do ombro."""
    q = m.qpos0.copy()
    for name, v in POSTURE.items():
        q[m.jnt_qposadr[m.joint(name).id]] = v
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        j1, j2 = m.eq_obj1id[i], m.eq_obj2id[i]
        a1, a2 = m.jnt_qposadr[j1], m.jnt_qposadr[j2]
        c = m.eq_data[i][:5]
        dx = q[a2] - m.qpos0[a2]
        q[a1] = m.qpos0[a1] + sum(c[k] * dx ** k for k in range(5))
    return q


def build_wrist_model(timestep=1e-3, tau_act=0.012, tau_deact=0.040,
                      contacts=False, free_joints=FREE_JOINTS, muscles=MUSCLES):
    """Compila o MjModel reduzido do punho."""
    # 1) posição de todos os corpos na postura desejada (theta = phi = 0)
    m_full = mujoco.MjModel.from_xml_path(MYOARM_XML)
    d_full = mujoco.MjData(m_full)
    d_full.qpos[:] = _posture_qpos(m_full)
    mujoco.mj_kinematics(m_full, d_full)

    spec = mujoco.MjSpec.from_file(MYOARM_XML)

    # 2) embute a pose relativa dos corpos cujas juntas serão removidas
    for b in spec.bodies:
        if b.name == "world":
            continue
        js = list(b.joints)
        if not js or all(j.name in free_joints for j in js):
            continue
        bid = m_full.body(b.name).id
        pid = m_full.body_parentid[bid]
        xp, xq = d_full.xpos[pid], d_full.xquat[pid]
        qinv = np.zeros(4)
        mujoco.mju_negQuat(qinv, xq)
        pos = np.zeros(3)
        mujoco.mju_rotVecQuat(pos, d_full.xpos[bid] - xp, qinv)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, qinv, d_full.xquat[bid])
        b.pos, b.quat = pos, quat
        b.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        for j in js:
            if j.name not in free_joints:
                spec.delete(j)

    # 3) remove acoplamentos (juntas do ombro não existem mais)
    for e in list(spec.equalities):
        spec.delete(e)

    # 4) mantém só os 7 músculos (e seus tendões)
    keep_tendons = set()
    for a in list(spec.actuators):
        if a.name in muscles:
            keep_tendons.add(a.target)
            a.dynprm[0], a.dynprm[1] = tau_act, tau_deact
        else:
            spec.delete(a)
    for t in list(spec.tendons):
        if t.name not in keep_tendons:
            spec.delete(t)
    for k in list(spec.keys):
        spec.delete(k)

    spec.option.timestep = timestep
    if not contacts:
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return spec.compile()


class WristPlant:
    """Interface simples de simulação: u (7 excitações) -> estados da tese."""

    def __init__(self, **kw):
        self.m = build_wrist_model(**kw)
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        m = self.m
        self.act_id = {n: m.actuator(n).id for n in MUSCLES}
        self.ctrl_idx = np.array([self.act_id[n] for n in CTRL_MUSCLES])
        self.state_act_idx = np.array([self.act_id[n] for n in STATE_MUSCLES])
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ["flexion", "pro_sup"]])
        self.vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ["flexion", "pro_sup"]])

    def reset(self, theta0=-0.1745, phi0=0.5236, a0=0.01):
        """Estado inicial (default_value do .osim: theta=-10°, phi=30°)."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = [theta0, phi0]
        self.d.act[:] = a0
        mujoco.mj_forward(self.m, self.d)
        return self.state()

    def state(self):
        """x = [asup aecrl afcu apq theta phi thetadot phidot] (rad, rad/s)."""
        d = self.d
        return np.concatenate([d.act[self.state_act_idx], d.qpos[self.qadr], d.qvel[self.vadr]])

    def excitation_vector(self, u4, others=0.01):
        """Monta as 7 excitações a partir de u=[ECRL FCU PQ SUP] (demais = 0.01)."""
        u = np.full(self.m.nu, others)
        u[self.ctrl_idx] = u4
        return np.clip(u, 0.0, 1.0)

    def step(self, u7):
        self.d.ctrl[:] = u7
        mujoco.mj_step(self.m, self.d)
        return self.state()


class WristPlantND:
    """
    Planta genérica (config "2dof" ou "3dof"); todos os músculos são comandados.
    x = [a (n_mus, na ordem de CONFIGS[..]['muscles']), q (n_q), qdot (n_q)]
    """

    def __init__(self, config="3dof", **kw):
        cfg = CONFIGS[config]
        self.muscles = list(cfg["muscles"])
        self.joints = list(cfg["free_joints"])
        self.coords = list(cfg["coord_names"])
        self.m = build_wrist_model(free_joints=self.joints, muscles=self.muscles, **kw)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.dt = m.opt.timestep
        self.n_mus, self.n_q = len(self.muscles), len(self.joints)
        self.act_idx = np.array([m.actuator(n).id for n in self.muscles])
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in self.joints])
        self.vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in self.joints])
        self.state_names = ([f"a_{n.lower()}" for n in self.muscles] + self.coords +
                            [c + "dot" for c in self.coords])
        self.iq = np.arange(self.n_mus, self.n_mus + self.n_q)          # índices de q em x
        self.iv = self.iq + self.n_q                                      # índices de qdot em x
        self.F0 = m.actuator_gainprm[self.act_idx, 2].copy()

    def reset(self, q0=None, a0=0.01):
        mujoco.mj_resetData(self.m, self.d)
        if q0 is not None:
            self.d.qpos[self.qadr] = q0
        self.d.act[:] = a0
        mujoco.mj_forward(self.m, self.d)
        return self.state()

    def state(self):
        d = self.d
        return np.concatenate([d.act[self.act_idx], d.qpos[self.qadr], d.qvel[self.vadr]])

    def moment_arms(self):
        """Matriz R (n_q x n_mus): torque [Nm] por N de tração, sinal = sentido +q."""
        m, d = self.m, self.d
        M = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(M, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        return -M[np.ix_(self.act_idx, self.vadr)].T

    def step(self, u):
        self.d.ctrl[self.act_idx] = np.clip(u, 0.0, 1.0)
        mujoco.mj_step(self.m, self.d)
        return self.state()


if __name__ == "__main__":
    import time
    p = WristPlant()
    m = p.m
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} ntendon={m.ntendon} dt={m.opt.timestep}")
    print("juntas:", [m.joint(i).name for i in range(m.njnt)])
    print("músculos:", [m.actuator(i).name for i in range(m.nu)])
    print("dynprm:", m.actuator_dynprm[:, :2][0])
    x = p.reset()
    u = p.excitation_vector(np.full(4, 0.01))
    t0 = time.perf_counter()
    for k in range(10000):
        x = p.step(u)
        if k % 2000 == 0:
            print(f"t={p.d.time:5.2f}  theta={np.rad2deg(x[4]):7.2f}°  phi={np.rad2deg(x[5]):7.2f}°")
    print(f"10 s simulados em {time.perf_counter()-t0:.3f} s")
