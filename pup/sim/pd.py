"""A motor driver computes torque from position and velocity errors."""

import mujoco
import numpy as np


class PDController:
    """Joint proportional-derivative control with physical torque limits."""

    def __init__(self, kp: np.ndarray | float, kd: np.ndarray | float,
                 torque_limit: float = 20.0) -> None:
        """Store scalar or (12,) gains in Nm/rad and Nm/(rad/s), limit in Nm."""
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

        return None

        raise NotImplementedError(
            "Stage 1: Store the PD gains and torque limit. See docs/01_mujoco_and_pd.md")
        # ===== end TODO =====

    def __call__(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray,
                 qd_des: np.ndarray | None = None) -> np.ndarray:
        """Return (12,) clipped torques, Nm, for (12,) angles/rates in rad/rad/s."""
        self.q = q
        self.qd = qd
        self.q_des = q_des
        self.qd_des = qd_des

        if type(self.qd_des) == type(None):
            torque = self.kp * (self.q_des - self.q) + self.kd * (-self.qd)
        else:
            self.qd_des[self.qd_des == None] = 0
            torque = self.kp * (self.q_des - self.q) + self.kd * (self.qd_des - self.qd)

        torque_clipped = np.clip(torque, -self.torque_limit, self.torque_limit)

        return torque_clipped

        raise NotImplementedError(
            "Stage 1: Calculate and clip joint torques. See docs/01_mujoco_and_pd.md")
        # ===== end TODO =====


def joint_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of actuated q (12,), rad, and qd (12,), rad/s; root is free."""

    q = data.qpos[7:].copy()
    qd = data.qvel[6:].copy()

    return (q, qd)

    raise NotImplementedError(
        "Stage 1: Slice joint positions and velocities past the free root. See docs/01_mujoco_and_pd.md")
    # ===== end TODO =====
