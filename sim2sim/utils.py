import os
import math
import torch
import random
import pickle
import datetime
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from legged_gym.utils.math import (
    normalize,
    euler_xyz_to_quat,
    quat_mul,
    quat_conjugate,
    quat_mul_yaw_inverse,
    quat_to_angle_axis,
    quat_to_euler_xyz,
)


def load_imitation_dataset(folder, mapping="joint_id.txt", suffix=".npz"):
    filenames = [name for name in os.listdir(folder) if name[-len(suffix):] == suffix]
    
    datatset = {}
    print("Loading motion dataset...")
    for filename in tqdm(filenames):
        try:
            data = pickle.load(open(os.path.join(folder, filename), 'rb'))
            datatset[filename[:-len(suffix)]] = data
        except:
            print(f"{filename} load failed!!!")
            continue
    
    dataset_names = list(datatset.keys())
    random.shuffle(dataset_names)
    dataset_list = [datatset[name] for name in dataset_names]
    
    lines = open(mapping).readlines()
    lines = [line[:-1].split(" ") for line in lines]
    joint_id_dict = {k: int(v) for v, k in lines}
    return dataset_list, dataset_names, joint_id_dict


def filter_legal_motion(datasets, data_names, base_height_range, base_roll_range, base_pitch_range, min_time):
    legal_datasets, legal_names, total_length, total_time = [], [], 0, 0.0
    
    print("Filtering motion dataset...")
    for data, name in zip(datasets, data_names):
        base_position = torch.tensor(data["base_position"], dtype=torch.float)
        base_orientation = torch.tensor(data["base_orientation"], dtype=torch.float)
        
        min_height_ids = (base_position[:, 2] < min(base_height_range)).nonzero().flatten()
        max_height_ids = (base_position[:, 2] > max(base_height_range)).nonzero().flatten()
        
        motion_base_quat = euler_xyz_to_quat(base_orientation)
        motion_base_rpy = quat_to_euler_xyz(quat_mul_yaw_inverse(motion_base_quat, motion_base_quat))
        
        min_base_roll_ids = (motion_base_rpy[:, 0] < min(base_roll_range)).nonzero().flatten()
        max_base_roll_ids = (motion_base_rpy[:, 0] > max(base_roll_range)).nonzero().flatten()
        
        min_base_pitch_ids = (motion_base_rpy[:, 1] < min(base_pitch_range)).nonzero().flatten()
        max_base_pitch_ids = (motion_base_rpy[:, 1] > max(base_pitch_range)).nonzero().flatten()
        
        illegal_id_list = torch.cat([
            min_height_ids, min_base_roll_ids, min_base_pitch_ids, 
            max_height_ids, max_base_roll_ids, max_base_pitch_ids,], dim=0)
        
        if len(illegal_id_list) > 0:
            first_illegal_ids = torch.amin(illegal_id_list).item()
            if first_illegal_ids > max(math.ceil(min_time * data["framerate"]), 3):
                data["base_position"] = data["base_position"][:first_illegal_ids]
                data["base_orientation"] = data["base_orientation"][:first_illegal_ids]
                data["joint_position"] = data["joint_position"][:first_illegal_ids]
                    
                for n in data["link_position"].keys():
                    data["link_position"][n] = data["link_position"][n][:first_illegal_ids]
                    data["link_orientation"][n] = data["link_orientation"][n][:first_illegal_ids]
                
                legal_datasets += [data]
                legal_names += [name]
                total_length += first_illegal_ids
                total_time += first_illegal_ids / data["framerate"]
        else:
            legal_datasets += [data]
            legal_names += [name]
            total_length += data["base_position"].shape[0]
            total_time += data["base_position"].shape[0] / data["framerate"]
    
    print("Number of legal motion dataset: ", len(legal_datasets))
    print("Total frame number: ", total_length)
    print("Total time: ", str(datetime.timedelta(seconds=total_time)))
    return legal_datasets, legal_names
        
        
class MotionLib:
    def __init__(self, datasets, data_names, mapping, dof_names, body_names, device="cpu"):
        self.device = device
        self.data_names = data_names
        
        self.fps = datasets[0]["framerate"]
        get_len = lambda x: x["base_position"].shape[0] - 1
        self.length = torch.tensor(
            [get_len(data) for data in datasets], dtype=torch.long, device=device)
        self.num_motion, self.total_length = self.length.shape[0], self.length.sum()
        
        self.random_indices = self.length.clone() - 1
        self.num_success = torch.zeros(self.num_motion, dtype=torch.float, device=device)
        
        self.end_indices = torch.cumsum(self.length, dim=0)
        self.start_indices = torch.nn.functional.pad(self.end_indices, (1, -1), "constant", 0)
        
        self.base_rpy = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_pos = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_lin_vel = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_ang_vel = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.dof_pos = torch.zeros(self.total_length, len(dof_names), dtype=torch.float, device=device)
        self.dof_vel = torch.zeros(self.total_length, len(dof_names), dtype=torch.float, device=device)
        self.body_pos = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_rpy = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_lin_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_ang_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        
        def compute_linear_velocity(xyz):
            return (xyz[1:] - xyz[:-1]) * self.fps
        
        def compute_angular_velocity(rpy):
            quat = euler_xyz_to_quat(rpy)
            diff_quat = quat_mul(quat[1:], quat_conjugate(quat[:-1]))
            angle, axis = quat_to_angle_axis(diff_quat)
            return axis * angle[..., None] * self.fps

        def compute_joint_velocity(q):
            return (q[1:] - q[:-1]) * self.fps
        
        print(f"Moving motion dataset to {self.device}...")
        for i, data in enumerate(tqdm(datasets)):
            start, end = self.start_indices[i], self.end_indices[i]
            base_pos = torch.tensor(data["base_position"], dtype=torch.float, device=device)
            base_rpy = torch.tensor(data["base_orientation"], dtype=torch.float, device=device)
                        
            self.base_pos[start:end] = base_pos[:self.length[i]]
            self.base_rpy[start:end] = base_rpy[:self.length[i]]
            self.base_lin_vel[start:end] = compute_linear_velocity(base_pos)[:self.length[i]]
            self.base_ang_vel[start:end] = compute_angular_velocity(base_rpy)[:self.length[i]]
            
            dof_pos = torch.tensor(data["joint_position"], dtype=torch.float, device=device)
            dof_vel = compute_joint_velocity(dof_pos)
            for j, name in enumerate(dof_names):
                if name in mapping.keys():
                    self.dof_pos[start:end, j] = dof_pos[:self.length[i], mapping[name]]
                    self.dof_vel[start:end, j] = dof_vel[:self.length[i], mapping[name]]

            for k, name in enumerate(body_names):
                body_pos = torch.tensor(data["link_position"][name], dtype=torch.float, device=device)
                body_rpy = torch.tensor(data["link_orientation"][name], dtype=torch.float, device=device)
                
                self.body_pos[start:end, k] = body_pos[:self.length[i]]
                self.body_rpy[start:end, k] = body_rpy[:self.length[i]]
                self.body_lin_vel[start:end, k] = compute_linear_velocity(body_pos)[:self.length[i]]
                self.body_ang_vel[start:end, k] = compute_angular_velocity(body_rpy)[:self.length[i]]

        # flush
        del datasets

    def check_success(self, motion_ids, motion_times):
        return torch.ceil(motion_times * self.fps) >= (self.length[motion_ids] - 1)
    
    def sample_motions(self, num_samples):
        max_num_success = torch.amax(torch.clip(self.num_success, min=1.0), dim=0)
        sampling_weight = torch.clip(1.0 - self.num_success / max_num_success, min=0.0)
        return torch.multinomial(sampling_weight + 1e-3, num_samples=num_samples, replacement=True)
            
    def sample_init_time(self, motion_ids, uniform=False):
        if uniform:
            phase = torch.rand(motion_ids.shape, dtype=torch.float, device=self.device)
            return torch.floor(phase * (self.length[motion_ids] - 1)) / self.fps
        return torch.zeros(motion_ids.shape, dtype=torch.float, device=self.device)

    def update_imitation_info(self, motion_ids, success, init_time):
        success_ids = success.nonzero().flatten()
        if len(success_ids) == 0: return
        
        success_motion_ids = motion_ids[success_ids]
        start_incides = torch.floor(init_time[success_ids] * self.fps).long()
        self.random_indices.scatter_reduce_(0, success_motion_ids, 
            start_incides, reduce="amin", include_self=True)
        self.num_success.scatter_reduce_(0, success_motion_ids,
            (start_incides == 0).float(), reduce="sum", include_self=True)

    def get_task_info(self):
        normalized_length = (self.length - 1) / (self.total_length - self.num_motion)
        completion = torch.sum(self.get_completion() * normalized_length)
        
        success_rate = (self.num_success > 0).float()
        success_rate = torch.sum(success_rate / self.num_motion)
        return completion, success_rate

    def get_motion_time(self, motion_ids):
        return (self.length[motion_ids] - 1) / self.fps
    
    def get_completion(self, motion_ids=None):
        completion = (self.length - 1 - self.random_indices) / (self.length - 1)
        return completion if motion_ids is None else completion[motion_ids]
        
    def get_motion_states(self, motion_ids, motion_times):
        timesteps = torch.minimum(motion_times * self.fps,
                                  self.length[motion_ids] - 2)
        floors = torch.floor(timesteps).long()
        
        motion_start_ids = self.start_indices[motion_ids]
        blend_motion_linear = lambda x: self.calc_blend(
            x, motion_start_ids + floors, timesteps - floors)
        blend_motion_slerp = lambda x: self.calc_slerp(
            x, motion_start_ids + floors, timesteps - floors)
                
        return dict(
            base_pos=blend_motion_linear(self.base_pos), 
            base_quat=blend_motion_slerp(self.base_rpy),
            base_lin_vel=blend_motion_linear(self.base_lin_vel),
            base_ang_vel=blend_motion_linear(self.base_ang_vel),
            
            dof_pos=blend_motion_linear(self.dof_pos), 
            dof_vel=blend_motion_linear(self.dof_vel),
            
            body_pos=blend_motion_linear(self.body_pos), 
            body_quat=blend_motion_slerp(self.body_rpy),
            body_lin_vel=blend_motion_linear(self.body_lin_vel), 
            body_ang_vel=blend_motion_linear(self.body_ang_vel),
            )
    
    @staticmethod    
    def calc_blend(motion, frame, t):
        motion0, motion1 = motion[frame], motion[frame + 1]
        shape = t.shape + (1,) * (motion0.dim() - t.dim())
        return t.view(*shape) * motion0 + (1 - t).view(*shape) * motion1

    @staticmethod    
    def calc_slerp(motion, frame, t):
        motion0 = euler_xyz_to_quat(motion[frame])
        motion1 = euler_xyz_to_quat(motion[frame + 1])
        
        shape = t.shape + (1,) * (motion0.dim() - t.dim())
        w0, w1 = t.view(*shape), (1 - t).view(*shape)
        
        cosine = F.cosine_similarity(motion0, motion1, dim=-1)
        cosine = cosine[..., None]
        motion1 = torch.where(cosine < 0.0, -motion1, motion1)
        
        cosine = torch.abs(cosine)
        theta = torch.clip(torch.acos(cosine), min=1e-3)
        sin_theta_recip = 1.0 / torch.sin(theta)

        s0 = torch.sin(theta * w0) * sin_theta_recip
        s1 = torch.sin(theta * w1) * sin_theta_recip
        return normalize(torch.where(cosine > 0.9995,
            w0 * motion0 + w1 * motion1, s0 * motion0 + s1 * motion1))
