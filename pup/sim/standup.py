"""Raise Pup smoothly from crouch before asking it to walk."""

from contextlib import nullcontext

import time
import mujoco
import mujoco.viewer
import numpy as np
import math

from pup.sim.pd import PDController, joint_state
from pup.sim.viewer import load_scene, reset_to_keyframe
from pup.envs.constants import JOINT_NAMES, DEFAULT_POSE, SCENE

def stand_up(duration_s: float = 3.0, headless: bool = True,
             kp: float = 40.0, kd: float = 1.0) -> dict:  # TODO(student): tune
    """Return final_height (m), max_roll/max_pitch (rad), and fell (bool).

    Interpolate (12,) target angles from crouch to home in one second;
    then hold until duration_s.

    The default gains above are the spring-2026 quadruped's (kp=10). Pup is
    heavier -- run it, watch it sag, and tune them (Stage 1, task 3). The
    test reads whatever defaults you leave in the signature.
    """
    #Initialize mujoco environment
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    

    CROUCH_POSE = np.tile([0, 1.4, -2.4], 4)
    delta = DEFAULT_POSE - CROUCH_POSE 

    controller = PDController(kp=kp, kd=kd)

    roll = []
    pitch = []
    trunk_z_list = []

    # Initialize robot in crouch pose
    data.qpos[7:] = CROUCH_POSE  
    data.qvel[:] = 0

    mujoco.mj_forward(model, data)

    #Run the pose interpolation with viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(int((1 + duration_s) / model.opt.timestep)): # Runs for 1.0s + duration_s
            step_start = time.time()

            current_joints = joint_state(model, data)
            q = current_joints[0]
            qd = current_joints[1]

            w, x, y, z = data.qpos[3:7]
            trunk_z = data.qpos[2]

            roll.append(np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)))
            pitch.append(np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0)))
            trunk_z_list.append(trunk_z)

            # Set joint targets to interpolated pose if time < 1.0s, and the crouch pose if not
            if data.time < 1.0:
                target = CROUCH_POSE +  delta * min(data.time / 1.0, 1.0)           
            else:
                target = DEFAULT_POSE
            
            # Use PD controller to set the torque based on joint targets
            data.ctrl = controller(q=q, qd=qd, q_des=target, qd_des=None)


            mujoco.mj_step(model, data) 

            viewer.sync() 

            # Make sure viewer runs in real time
            time_remaining = (
            model.opt.timestep - (time.time() - step_start)
            )

            if time_remaining > 0:
                time.sleep(time_remaining)


    fell = any(
    (not math.isfinite(z)) or (z < 0.12)
    for z in trunk_z_list
    )
    
    return {"final_height": trunk_z_list[-1],   
     "max_roll": max(roll),       
     "max_pitch": max(pitch),      
     "fell": fell}

    # ===== TODO(student): Interpolate from crouch to home and measure stability =====
    # ===== end TODO =====
