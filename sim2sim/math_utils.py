# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import math
import torch
import torch.nn.functional as F

# @torch.jit.script
def align_shape(a, b):
    assert a.dim() == b.dim()
    if sum(a.shape[:-1]) > sum(b.shape[:-1]):
        zero_mask = a[..., 0:1] * 0.0
    else:
        zero_mask = b[..., 0:1] * 0.0
    return a + zero_mask, b + zero_mask

# @torch.jit.script
def normalize(x, eps=1e-9):
    return F.normalize(x, p=2.0, dim=-1, eps=eps)

# @torch.jit.script
def copysign(a, b):
    return torch.abs(a + b * 0.0) * torch.sign(b)

# @torch.jit.script
def quat_conjugate(x):
    return torch.cat([-x[..., :3], x[..., 3:]], dim=-1)

# @torch.jit.script
def quat_mul(a, b):
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1)

# @torch.jit.script
def quat_mul_inverse(a, b):
    return quat_mul(quat_conjugate(a), b)

# @torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

# @torch.jit.script
def quat_to_euler_xyz(q):
    q = normalize(q)
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
                q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(math.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
                q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return wrap_to_pi(torch.stack((roll, pitch, yaw), dim=-1))

# @torch.jit.script
def get_quat_yaw(quat):
    quat_yaw = quat.clone()
    quat_yaw[..., 0:2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_yaw

# @torch.jit.script
def quat_rotate_yaw(quat, vec):
    quat_yaw = get_quat_yaw(quat)
    return quat_rotate(quat_yaw, vec)

# @torch.jit.script
def quat_rotate_yaw_inverse(quat, vec):
    quat_yaw = get_quat_yaw(quat_conjugate(quat))
    return quat_rotate(quat_yaw, vec)

# @torch.jit.script
def quat_mul_yaw(a, b):
    return quat_mul(get_quat_yaw(a), b)

# @torch.jit.script
def quat_mul_yaw_inverse(a, b):
    return quat_mul(get_quat_yaw(quat_conjugate(a)), b)

# @torch.jit.script
def wrap_to_pi(angles):
    angles %= 2.0 * math.pi
    angles = torch.where(
        angles > math.pi, 
        angles - 2.0 * math.pi, angles)
    return angles

# @torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    r = 2.0 * torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.0) / 2.0
    return (upper - lower) * r + lower

# @torch.jit.script
def quat_rotate(q, v):
    q, v = align_shape(q, v)
    q_vec, q_w = q[..., 0:3], q[..., 3:4]
    a = v * (2.0 * torch.square(q_w) - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    dot = torch.einsum("...i,...i->...", q_vec, v)
    c = q_vec * dot[..., None] * 2.0
    return a + b + c

# @torch.jit.script
def quat_rotate_inverse(q, v):
    q, v = align_shape(q, v)
    q_vec, q_w = q[..., 0:3], q[..., 3:4]
    a = v * (2.0 * torch.square(q_w) - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    dot = q_vec[..., None, :] @ v[..., None]
    c = q_vec * dot.squeeze(-1) * 2.0
    return a - b + c

# @torch.jit.script
def quat_to_angle_axis(q):
    # computes axis-angle representation from quaternion q
    # q must be normalized
    q = normalize(q)
    sin_theta = torch.sqrt(1 - q[..., 3] * q[..., 3])
    angle = 2.0 * torch.acos(q[..., 3])
    angle = normalize_angle(angle)
    axis = q[..., 0:3] / sin_theta[..., None]

    mask = torch.abs(sin_theta) > 1e-5
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    axis = torch.where(mask[..., None], axis, default_axis)
    return wrap_to_pi(angle), axis

# @torch.jit.script
def angle_axis_to_quat(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([xyz, w], dim=-1))

# @torch.jit.script
def euler_xyz_to_quat(xyz):
    roll, pitch, yaw = torch.unbind(xyz, dim=-1)
        
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return normalize(q)

# @torch.jit.script
def quat_to_tan_norm(q):
    q = normalize(q)
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=-1)
    return norm_tan

# @torch.jit.script
def heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

# @torch.jit.script
def heading_quat(q):
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    return angle_axis_to_quat(heading(q), axis)

# @torch.jit.script
def heading_quat_conjugate(q):
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    return angle_axis_to_quat(-heading(q), axis)

# @torch.jit.script
def remove_heading_quat(q):
    heading_q = heading_quat_conjugate(q)
    return quat_mul(heading_q, q)

# @torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower

# @torch.jit.script
def torch_rand_like_float(lower, upper, tensor):
    return (upper - lower) * torch.rand_like(tensor) + lower