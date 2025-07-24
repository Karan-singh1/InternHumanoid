#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
from functools import reduce
from .normalization import Normalization
from .network import (
    HyperEmbedder, 
    HyperLERPBlock, 
    HyperPredictHead,
    GRUCell, 
    LSTMCell, 
    MinRNNCell,
    )


class Estimator(nn.Module):
    is_recurrent = False
    
    def __init__(
        self, 
        num_proprio_obs,
        num_estimations,
        estimator_hidden_dims=[256, 256],
        **kwargs,
    ):
        if kwargs:
            print(
                "Estimator.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()])
                  )
        
        super().__init__()
        mlp_input_dim_e = num_proprio_obs
        
        self.normalizers = nn.ModuleDict({
            "estimator": Normalization(shape=[num_proprio_obs]),
        })
        
        num_blocks = len(estimator_hidden_dims)
        estimator_hidden_dim = reduce(math.gcd, estimator_hidden_dims)
        estimator_layers = [
            HyperEmbedder(
                mlp_input_dim_e, estimator_hidden_dim,
                init=math.sqrt(2.0 / estimator_hidden_dim),
                scale=math.sqrt(2.0 / estimator_hidden_dim),
            )]
        for index in range(len(estimator_hidden_dims) - 1):
            dimensions = math.sqrt(estimator_hidden_dims[index])
            estimator_layers += [
                HyperLERPBlock(
                    estimator_hidden_dim,
                    estimator_hidden_dims[index] // estimator_hidden_dim,
                    scaler_init=math.sqrt(2.0 / estimator_hidden_dim),
                    scaler_scale=math.sqrt(2.0 / estimator_hidden_dim),
                    alpha_init=1.0 / num_blocks,
                    alpha_scale=1.0 / dimensions,
                    )]
        estimator_layers += [HyperPredictHead(estimator_hidden_dims[-1], num_estimations)]
        self.estimator = nn.Sequential(*estimator_layers)
        
    def reset(self, dones=None):
        return

    def forward(self):
        raise NotImplementedError

    def predict(self, observations, **kwargs):
        observations = self.normalizers["estimator"](observations)
        estimations = self.estimator(observations)
        return estimations
    
    def predict_inference(self, observations, **kwargs):
        observations = self.normalizers["estimator"](observations)
        estimations = self.estimator(observations)
        return estimations, None
            
    def get_hidden_states(self):
        return


class EstimatorRecurrent(Estimator):
    is_recurrent = True
    
    def __init__(
        self, 
        num_proprio_obs,
        num_estimations,
        estimator_hidden_dims=[256, 256],
        rnn_type="gru",
        rnn_hidden_dim=256,
        **kwargs,
    ):
        if kwargs:
            print(
                "EstimatorRecurrent.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()])
                  )
        
        super().__init__(
            num_proprio_obs=rnn_hidden_dim,
            num_estimations=num_estimations,
            estimator_hidden_dims=estimator_hidden_dims,
        )
        
        self.normalizers.update({
            "memory_e": Normalization(shape=[num_proprio_obs]),
            })
        
        self.rnn_type, self.rnn_hidden_dim = rnn_type, rnn_hidden_dim
        if self.rnn_type.lower() == "gru":
            rnn_class = GRUCell
        elif self.rnn_type.lower() == "lstm":
            rnn_class = LSTMCell
        elif self.rnn_type.lower() == "minrnn":
            rnn_class = MinRNNCell
        else: raise ValueError(f"Invalid RNN type: {self.rnn_type.lower()}")
        
        self.memory_e = rnn_class(num_proprio_obs, rnn_hidden_dim)
        self.hidden_states_e = None

    def forward(self):
        raise NotImplementedError

    def predict(self, observations, hidden_states=None, **kwargs):
        memories, next_hidden_states = self.predict_memory(observations, hidden_states)
        return super().predict(memories)
    
    def predict_inference(self, observations, hidden_states=None, **kwargs):
        memories, next_hidden_states = self.predict_memory(observations, hidden_states)
        return super().predict(memories), next_hidden_states
    
    def predict_memory(self, observations, hidden_states=None):
        observations = self.normalizers["memory_e"](observations)
        if hidden_states is None:
            current_hidden_state = self.hidden_states_e
        else:
            current_hidden_state = hidden_states
        output, next_hidden_states = self.memory_e(observations, current_hidden_state)
        if hidden_states is None:
            self.hidden_states_e = next_hidden_states
        return output, next_hidden_states

    def reset(self, dones=None):
        assert dones is not None
        if isinstance(self.hidden_states_e, tuple):
            self.hidden_states_e[0][dones] = 0.0
            self.hidden_states_e[1][dones] = 0.0
        else:
            self.hidden_states_e[dones] = 0.0
            
    def get_hidden_states(self):
        if self.hidden_states_e is None:
            return self.hidden_states_e
        
        if isinstance(self.hidden_states_e, tuple):
            return self.hidden_states_e[0], self.hidden_states_e[1]
        else:
            return self.hidden_states_e