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
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil


class AxesGeometry(gymutil.LineGeometry):
    def __init__(self, scale=1.0, pose=None):
        verts = np.empty((3, 2), gymapi.Vec3.dtype)
        verts[0][0] = (0, 0, 0)
        verts[0][1] = (scale, 0, 0)
        verts[1][0] = (0, 0, 0)
        verts[1][1] = (0, scale, 0)
        verts[2][0] = (0, 0, 0)
        verts[2][1] = (0, 0, scale)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(3, gymapi.Vec3.dtype)
        colors[0] = (1.0, 0.0, 0.0)
        colors[1] = (0.0, 1.0, 0.0)
        colors[2] = (0.0, 0.0, 1.0)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class WireframeBoxGeometry(gymutil.LineGeometry):
    def __init__(self, xdim=1, ydim=1, zdim=1, pose=None, color=None):
        if color is None:
            color = (1, 0, 0)

        num_lines = 12

        x = 0.5 * xdim
        y = 0.5 * ydim
        z = 0.5 * zdim

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        # top face
        verts[0][0] = (x, y, z)
        verts[0][1] = (x, y, -z)
        verts[1][0] = (-x, y, z)
        verts[1][1] = (-x, y, -z)
        verts[2][0] = (x, y, z)
        verts[2][1] = (-x, y, z)
        verts[3][0] = (x, y, -z)
        verts[3][1] = (-x, y, -z)
        # bottom face
        verts[4][0] = (x, -y, z)
        verts[4][1] = (x, -y, -z)
        verts[5][0] = (-x, -y, z)
        verts[5][1] = (-x, -y, -z)
        verts[6][0] = (x, -y, z)
        verts[6][1] = (-x, -y, z)
        verts[7][0] = (x, -y, -z)
        verts[7][1] = (-x, -y, -z)
        # verticals
        verts[8][0] = (x, y, z)
        verts[8][1] = (x, -y, z)
        verts[9][0] = (x, y, -z)
        verts[9][1] = (x, -y, -z)
        verts[10][0] = (-x, y, z)
        verts[10][1] = (-x, -y, z)
        verts[11][0] = (-x, y, -z)
        verts[11][1] = (-x, -y, -z)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class WireframeBBoxGeometry(gymutil.LineGeometry):

    def __init__(self, bbox, pose=None, color=None):
        if bbox.shape != (2, 3):
            raise ValueError('Expected bbox to be a matrix of 2 by 3!')

        if color is None:
            color = (1, 0, 0)

        num_lines = 12

        min_x, min_y, min_z = bbox[0]
        max_x, max_y, max_z = bbox[1]

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        # top face
        verts[0][0] = (max_x, max_y, max_z)
        verts[0][1] = (max_x, max_y, min_z)
        verts[1][0] = (min_x, max_y, max_z)
        verts[1][1] = (min_x, max_y, min_z)
        verts[2][0] = (max_x, max_y, max_z)
        verts[2][1] = (min_x, max_y, max_z)
        verts[3][0] = (max_x, max_y, min_z)
        verts[3][1] = (min_x, max_y, min_z)

        # bottom face
        verts[4][0] = (max_x, min_y, max_z)
        verts[4][1] = (max_x, min_y, min_z)
        verts[5][0] = (min_x, min_y, max_z)
        verts[5][1] = (min_x, min_y, min_z)
        verts[6][0] = (max_x, min_y, max_z)
        verts[6][1] = (min_x, min_y, max_z)
        verts[7][0] = (max_x, min_y, min_z)
        verts[7][1] = (min_x, min_y, min_z)

        # verticals
        verts[8][0] = (max_x, max_y, max_z)
        verts[8][1] = (max_x, min_y, max_z)
        verts[9][0] = (max_x, max_y, min_z)
        verts[9][1] = (max_x, min_y, min_z)
        verts[10][0] = (min_x, max_y, max_z)
        verts[10][1] = (min_x, min_y, max_z)
        verts[11][0] = (min_x, max_y, min_z)
        verts[11][1] = (min_x, min_y, min_z)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class WireframeSphereGeometry(gymutil.LineGeometry):

    def __init__(self, radius=1.0, num_lats=8, num_lons=8, pose=None, color=None, color2=None):
        if color is None:
            color = (1, 0, 0)

        if color2 is None:
            color2 = color

        num_lines = 2 * num_lats * num_lons

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        idx = 0

        ustep = 2 * math.pi / num_lats
        vstep = math.pi / num_lons

        u = 0.0
        for i in range(num_lats):
            v = 0.0
            for j in range(num_lons):
                x1 = radius * math.sin(v) * math.sin(u)
                y1 = radius * math.cos(v)
                z1 = radius * math.sin(v) * math.cos(u)

                x2 = radius * math.sin(v + vstep) * math.sin(u)
                y2 = radius * math.cos(v + vstep)
                z2 = radius * math.sin(v + vstep) * math.cos(u)

                x3 = radius * math.sin(v + vstep) * math.sin(u + ustep)
                y3 = radius * math.cos(v + vstep)
                z3 = radius * math.sin(v + vstep) * math.cos(u + ustep)

                verts[idx][0] = (x1, y1, z1)
                verts[idx][1] = (x2, y2, z2)
                colors[idx] = color

                idx += 1

                verts[idx][0] = (x2, y2, z2)
                verts[idx][1] = (x3, y3, z3)
                colors[idx] = color2

                idx += 1

                v += vstep
            u += ustep

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors
    

class WireframeArrowGeometry(gymutil.LineGeometry):
    def __init__(self, direction, length=1.0, head_length=0.2, head_width=0.1, shaft_radius=0.04, shaft_segments=16, pose=None, color=None, color2=None):
        if color is None:
            color = (1, 0, 0)
 
        if color2 is None:
            color2 = color
 
        direction = np.array(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        # Calculate main arrow shaft endpoint
        shaft_end_point = length * direction
        arrow_tip = (length + head_length) * direction
 
        # Arrow shaft
        verts = []
        colors = []
 
        # Generate perpendicular vectors to direction for shaft
        perp1 = np.cross(direction, np.array([1, 0, 0]))
        if np.linalg.norm(perp1) < 1e-6:
            perp1 = np.cross(direction, np.array([0, 1, 0]))
        perp1 = perp1 / np.linalg.norm(perp1) * shaft_radius
 
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2) * shaft_radius
 
        # Generate shaft lines in a circular pattern
        angle_step = 2 * math.pi / shaft_segments
        shaft_base_points = []
        arrow_base_points = []
 
        for i in range(shaft_segments):
            angle = i * angle_step
            next_angle = (i + 1) * angle_step
 
            offset1 = math.cos(angle) * perp1 + math.sin(angle) * perp2
            offset2 = math.cos(next_angle) * perp1 + math.sin(next_angle) * perp2
 
            start_circle = offset1
            end_circle = shaft_end_point + offset1
            shaft_base_points.append(end_circle)
 
            verts.append((start_circle, end_circle))
            colors.append(color)
 
            verts.append((start_circle, offset2))
            colors.append(color)
 
            verts.append((end_circle, shaft_end_point + offset2))
            colors.append(color)
 
        # Arrow head base point
        arrow_base = shaft_end_point
 
        # Generate perpendicular vectors to direction for arrow head
        perp1_head = perp1 / shaft_radius * head_width
        perp2_head = perp2 / shaft_radius * head_width
 
        # Generate arrow head lines to represent a cone
        for i in range(shaft_segments):
            angle = i * angle_step
            next_angle = (i + 1) * angle_step
 
            offset1 = math.cos(angle) * perp1_head + math.sin(angle) * perp2_head
            offset2 = math.cos(next_angle) * perp1_head + math.sin(next_angle) * perp2_head
 
            base_point1 = arrow_base + offset1
            base_point2 = arrow_base + offset2
            arrow_base_points.append(base_point1)
 
            # Lines from tip to base circle
            verts.append((arrow_tip, base_point1))
            colors.append(color2)
 
            # Lines around the base circle
            verts.append((base_point1, base_point2))
            colors.append(color2)
 
        # Connect corresponding points on the shaft end and arrow base
        for shaft_point, arrow_point in zip(shaft_base_points, arrow_base_points):
            verts.append((shaft_point, arrow_point))
            colors.append(color2)
 
        # Convert verts and colors to numpy arrays
        num_lines = len(verts)
        verts_np = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        colors_np = np.empty(num_lines, gymapi.Vec3.dtype)
 
        for idx, (v_start, v_end) in enumerate(verts):
            verts_np[idx][0] = (v_start[0], v_start[1], v_start[2])
            verts_np[idx][1] = (v_end[0], v_end[1], v_end[2])
            colors_np[idx] = colors[idx]
 
        # Apply pose transformation if provided
        if pose is None:
            self.verts = verts_np
        else:
            self.verts = pose.transform_points(verts_np)
 
        self._colors = colors_np
 
    def vertices(self):
        return self.verts
 
    def colors(self):
        return self._colors