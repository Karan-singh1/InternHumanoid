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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
import torch
import hydra
from easydict import EasyDict
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, load_onnx_policy
from rsl_rl.runners import OnPolicyRunner
from omegaconf import OmegaConf, DictConfig

@hydra.main(version_base=None,config_path="../config", config_name="base")
def train(cfg: OmegaConf):

    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    args = get_args(cfg.args)
    cfg.pop('args')
    
    env_cfg, train_cfg = EasyDict(), EasyDict()
    for key in cfg.keys():
        train_cfg.update(cfg[key]) if key == 'algo' else env_cfg.update(cfg[key])

    if not args.headless:
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
    
    env, env_cfg = task_registry.make_env(name=env_cfg.name, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_runner(env=env, runner_class=OnPolicyRunner, name=env_cfg.name, args=args, train_cfg=train_cfg)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)



if __name__ == '__main__':
    train()
