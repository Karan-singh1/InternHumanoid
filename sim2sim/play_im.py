import os
import time
import torch
import mujoco
import pickle
import argparse
import numpy as np
import onnxruntime
import mujoco.viewer
from collections import deque
from legged_gym import LEGGED_GYM_ROOT_DIR
from math_utils import quat_rotate_yaw, quat_rotate_yaw_inverse, quat_to_tan_norm
from utils import load_imitation_dataset, filter_legal_motion, MotionLib


SIMULATION_DURATION = 300.0 # sec
SIMULATION_dt = 0.0025 # sec
CONTROL_DECIMATION = 10 # steps


FUTURE_DT = 0.1  # sec
FUTURE_TIMESTEPS = 5
HISTORY_TIMESTEPS = 5


def xyzw_to_wxyz(quat):
    return torch.roll(quat, 1, dims=-1)


def wxyz_to_xyzw(quat):
    return torch.roll(quat, -1, dims=-1)


def get_projected_gravity(quat):
    vec = np.array([0.0, 0.0, -1.0])
    q_w, q_vec = quat[0], quat[1:4]
    a = vec * (2.0 * np.square(q_w) - 1.0)
    b = np.cross(q_vec, vec) * q_w * 2.0
    c = q_vec * np.dot(q_vec, vec) * 2.0
    return a - b + c


def compute_torques(target_dof_pos, dof_pos, dof_vel, kp, kd):
    """Calculates torques from position commands"""
    return (target_dof_pos - dof_pos) * kp - dof_vel * kd


def load_onnx_policy(path, device="cuda:0"):
    model = onnxruntime.InferenceSession(path)
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device=device)
    return run_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, help="robot name", required=True)
    args = parser.parse_args()
    
    # NOTE: need to change the xml path and ckpt path
    if args.robot == "g1_29dof":
        xml_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/G1/urdf/g1_29dof.xml"
        ckpt_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/g1_29dof_im.onnx"
        info_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/g1_29dof_im.pkl"
        prefix = f"{LEGGED_GYM_ROOT_DIR}/resources/dataset/g1_29dof_data"
    elif args.robot == "g1_23dof":
        xml_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/G1/urdf/g1_23dof.xml"
        ckpt_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/g1_23dof_im.onnx"
        info_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/g1_23dof_im.pkl"
        prefix = f"{LEGGED_GYM_ROOT_DIR}/resources/dataset/g1_23dof_data"
    elif args.robot == "h1":
        # NOTE: need to change the xml path and ckpt path
        xml_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/H1/urdf/h1.xml"
        ckpt_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/h1_im.onnx"
        info_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/h1_im.pkl"
        prefix = f"{LEGGED_GYM_ROOT_DIR}/resources/dataset/h1_data"
    elif args.robot == "h1_2":
        # NOTE: need to change the xml path and ckpt path
        xml_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/H1_2/urdf/h1_2.xml"
        ckpt_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/h1_2_im.onnx"
        info_path = f"{LEGGED_GYM_ROOT_DIR}/logs/export_model/h1_2_im.pkl"
        prefix = f"{LEGGED_GYM_ROOT_DIR}/resources/dataset/h1_2_data"
    else:
        raise ValueError(f"Invalid robot name: {args.robot}")
    
    # Load robot model and policy
    info = pickle.load(open(info_path, "rb"))
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_DT
    policy = load_onnx_policy(ckpt_path, device="cpu")
    
    # Load config
    STIFFNESS = info["STIFFNESS"]
    DAMPING = info["DAMPING"]
    ACTION_SCALE = info["ACTION SCALE"]
    DEFAULT_JOINT_ANGLES = info["DEFAULT JOINT ANGLES"]

    DOF_NAMES = info["DOF NAMES"]
    KEYFRAME_NAMES = info["KEYFRAME NAMES"]
    MOTION_DOF_INDICES = info["MOTION DOF INDICES"]
    ACTION_INDICES = info["ACTION INDICES"]
    DEFAULT_DOF_POS = info["DEFAULT DOF POS"]
    TORQUE_LIMITS = info["TORQUE LIMITS"]
    END_EFFECTOR_INDICES = info["END EFFECTOR INDICES"]
    
    tan_norm_scale = 1.0
    base_pos_scale = 1.0
    body_pos_scale = 1.0
    base_lin_vel_scale = 2.0
    base_ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    gravity_scale = 1.0
    height_points_scale = 5.0

    target_dof_pos = np.zeros(model.njnt - 1)
    default_dof_pos = np.zeros(model.njnt - 1)
    kp_gains = np.zeros(model.njnt - 1)
    kd_gains = np.zeros(model.njnt - 1)
    actions = np.zeros(model.njnt - 1)
    
    mujoco_joint_names = []
    for i in range(1, model.njnt):
        joint_name = model.joint(i).name
        mujoco_joint_names.append(joint_name)
        
        for k, v in DEFAULT_JOINT_ANGLES.items():
            if k in joint_name: default_dof_pos[i - 1] = v
                
        for k, v in STIFFNESS.items():
            if k in joint_name: kp_gains[i - 1] = v
        
        for k, v in DAMPING.items():
            if k in joint_name: kd_gains[i - 1] = v
    
    joint_reindex = [mujoco_joint_names.index(name) for name in DOF_NAMES]
    action_reindex = [mujoco_joint_names.index(DOF_NAMES[i]) for i in ACTION_INDICES]
    torque_limits = np.array([TORQUE_LIMITS[DOF_NAMES.index(name)] for name in mujoco_joint_names])
    motion_dof_pos_offset = torch.tensor(DEFAULT_DOF_POS, dtype=torch.float)

    # Load motion
    folder = "full_valid"
    joint_mapping = "joint_id.txt"
    
    min_time = 2.0 # sec
    base_height_range = [0.3, 1.5] # m
    base_roll_range = [-0.8, 0.8] # rad
    base_pitch_range = [-0.8, 1.2] # rad
    
    folder_path = os.path.join(prefix, folder)
    mapping_path = os.path.join(folder_path, joint_mapping)
    dataset, data_names, mapping = load_imitation_dataset(folder_path, mapping_path)
    dataset, data_names = filter_legal_motion(dataset, data_names, base_height_range,
        base_roll_range, base_pitch_range, min_time)
    motions = MotionLib(dataset, data_names, mapping, DOF_NAMES, KEYFRAME_NAMES)
    
    # reset motion & observation
    motion_ids = motions.sample_motions(1)
    motion_name = data_names[motion_ids.item()]
    print(f"Selected motion name: {motion_name}")
    motion_ids = motion_ids[:, None].repeat(1, FUTURE_TIMESTEPS)
    
    orders = torch.linspace(0, (FUTURE_TIMESTEPS - 1) * FUTURE_DT, FUTURE_TIMESTEPS, dtype=torch.float)
    motion_init_time = motions.sample_init_time(motion_ids[:, 0], uniform=False)
    motion_time = motion_init_time[:, None] + orders
    motion_total_time = motions.get_motion_time(motion_ids[0, 0])
    motion_dict = motions.get_motion_states(motion_ids, motion_time)
    
    data.qpos[0:3] = motion_dict["base_pos"][0, 0].numpy().astype(data.qpos.dtype)
    data.qpos[3:7] = xyzw_to_wxyz(motion_dict["base_quat"][0, 0]).numpy().astype(data.qpos.dtype)
    data.qpos[7:] = motion_dict["dof_pos"][0, 0].numpy().astype(data.qpos.dtype)[joint_reindex]
    data.qvel[:] = 0.0
    
    history_buffer = deque(maxlen=HISTORY_TIMESTEPS)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        counter = 0        
        start = time.time()
        while viewer.is_running() and time.time() - start < SIMULATION_DURATION:
            step_start = time.time()
            
            # NOTE: use float64 array to compute torques
            torques = compute_torques(target_dof_pos, data.qpos[7:], data.qvel[6:], kp_gains, kd_gains)
            data.ctrl[:] = np.clip(torques, a_min=-torque_limits, a_max=torque_limits)

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            if counter % CONTROL_DECIMATION == 0:
                motion_time = motion_time + SIMULATION_DT * CONTROL_DECIMATION
                motion_dict = motions.get_motion_states(motion_ids, motion_time)
                
                # reset
                timeout = motion_time[0, 0].item() > motion_total_time.item()
                if timeout:
                    motion_ids = motions.sample_motions(1)
                    motion_name = data_names[motion_ids.item()]
                    print(f"Selected motion name: {motion_name}")
                    
                    motion_ids = motion_ids[:, None].repeat(1, FUTURE_TIMESTEPS)
                    motion_init_time = motions.sample_init_time(motion_ids[:, 0], uniform=False)
                    motion_time = motion_init_time[:, None] + orders
                    motion_total_time = motions.get_motion_time(motion_ids[0, 0])
                    motion_dict = motions.get_motion_states(motion_ids, motion_time)
                
                fall_down = np.any(np.abs(get_projected_gravity(data.qpos[3:7])[0:2]) > 0.8)
                if fall_down:
                    motion_ids = motions.sample_motions(1)
                    motion_name = data_names[motion_ids.item()]
                    print(f"Selected motion name: {motion_name}")
                    
                    motion_ids = motion_ids[:, None].repeat(1, FUTURE_TIMESTEPS)
                    motion_init_time = motions.sample_init_time(motion_ids[:, 0], uniform=False)
                    motion_time = motion_init_time[:, None] + orders
                    motion_total_time = motions.get_motion_time(motion_ids[0, 0])
                    motion_dict = motions.get_motion_states(motion_ids, motion_time)
                    
                    data.qpos[0:3] = motion_dict["base_pos"][0, 0].numpy().astype(data.qpos.dtype)
                    data.qpos[3:7] = xyzw_to_wxyz(motion_dict["base_quat"][0, 0]).numpy().astype(data.qpos.dtype)
                    data.qpos[7:] = motion_dict["dof_pos"][0, 0].numpy().astype(data.qpos.dtype)[joint_reindex]
                    data.qvel[:] = 0.0
                    
                    history_buffer = deque(maxlen=HISTORY_TIMESTEPS)
                    
                # motion observations
                motion_base_tan_norm = quat_to_tan_norm(motion_dict["base_quat"]).view(1, -1)
                motion_base_lin_vel = quat_rotate_yaw_inverse(
                    motion_dict["base_quat"], motion_dict["base_lin_vel"]).view(1, -1)
                motion_base_ang_vel = quat_rotate_yaw_inverse(
                    motion_dict["base_quat"], motion_dict["base_ang_vel"]).view(1, -1)
                motion_dof_pos = (motion_dict["dof_pos"] - 
                                  motion_dof_pos_offset)[:, :, MOTION_DOF_INDICES].view(1, -1)
                motion_body_pos = quat_rotate_yaw_inverse(
                    motion_dict["base_quat"][..., None, :], 
                    motion_dict["body_pos"] - motion_dict["base_pos"][..., None, :])
                motion_body_pos[..., 2] = motion_dict["body_pos"][..., 2]
                motion_endeffector_pos = motion_body_pos[:, :, END_EFFECTOR_INDICES].view(1, -1)
                motion_time_phase = motion_time[0, 0:1] / motion_total_time
                
                # proprioceptive observations
                tensor_base_quat = wxyz_to_xyzw(torch.tensor(data.qpos[3:7], dtype=torch.float))
                gravity = get_projected_gravity(data.qpos[3:7])
                base_ang_vel = data.qvel[3:6].copy()
                dof_pos = (data.qpos[7:].copy() - default_dof_pos)[action_reindex]
                dof_vel = data.qvel[6:].copy()[action_reindex]
                last_action = actions[action_reindex]
                
                step_obs = np.concatenate([
                    gravity * gravity_scale,
                    base_ang_vel * base_ang_vel_scale,
                    dof_pos * dof_pos_scale,
                    dof_vel * dof_vel_scale,
                    last_action,
                    # overfitting
                    # motion_time_phase.numpy().astype(actions.dtype),
                    ], axis=0)[np.newaxis]

                if len(history_buffer) < HISTORY_TIMESTEPS:
                    history_buffer.extend([
                        torch.zeros(*step_obs.shape, dtype=torch.float)
                        ] * HISTORY_TIMESTEPS)
                    
                history_buffer.append(torch.tensor(step_obs, dtype=torch.float))
                pripo_obs_tensor = torch.cat(list(history_buffer), dim=1)
                
                obs_tensor = torch.cat(list(history_buffer) + [
                    motion_base_tan_norm * tan_norm_scale, 
                    motion_base_lin_vel * base_lin_vel_scale, 
                    motion_base_ang_vel * base_ang_vel_scale, 
                    motion_dof_pos * dof_pos_scale, 
                    motion_endeffector_pos * body_pos_scale,
                    ], dim=1)
                
                # policy inference
                policy_actions = policy(obs_tensor).squeeze()
                actions[action_reindex] = policy_actions.numpy().astype(actions.dtype)
                # transform action to target_dof_pos
                target_dof_pos = default_dof_pos + actions * ACTION_SCALE
                
                # visualize marker
                local_body_pos = motion_dict["body_pos"][:, 0] - motion_dict["base_pos"][:, 0:1]
                local_body_pos = quat_rotate_yaw_inverse(motion_dict["base_quat"][:, 0:1], local_body_pos)
                marker_pos = quat_rotate_yaw(tensor_base_quat[None], local_body_pos.view(-1, 3))
                marker_pos = marker_pos.numpy().astype(data.qpos.dtype) + data.qpos[0:3]
                viewer.user_scn.ngeom = 0
                for i in range(marker_pos.shape[0]):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0, 0],
                        pos=marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1])  
                    )
                viewer.user_scn.ngeom = marker_pos.shape[0]
            
            counter += 1
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0: time.sleep(time_until_next_step)