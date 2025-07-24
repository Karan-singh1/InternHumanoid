#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(self, num_envs, num_transitions, state_shape, device="cpu"):
        self.states = torch.zeros(num_transitions, num_envs, *state_shape, dtype=torch.float)
        self.next_states = torch.zeros(num_transitions, num_envs, *state_shape, dtype=torch.float)
        
        self.step = 0
        self.num_samples = 0
        self.device = device
        self.num_envs = num_envs
        self.num_transitions = num_transitions
    
    def add_transitions(self, states, next_states):
        self.states[self.step].copy_(states)
        self.next_states[self.step].copy_(next_states)
        self.step = self.step % self.num_transitions
        self.num_samples = min(self.num_samples + 1, self.num_transitions)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8, mini_batch_size=512):        
        states = self.states.flatten(0, 1)
        next_states = self.next_states.flatten(0, 1)
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                indices = torch.randint(self.num_samples * self.num_envs, (mini_batch_size,))
                state_batch = states[indices].to(self.device)
                next_state_batch = next_states[indices].to(self.device)
                yield state_batch, next_state_batch