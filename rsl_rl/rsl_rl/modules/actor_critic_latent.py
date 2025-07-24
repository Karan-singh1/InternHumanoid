#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from torch.distributions import Normal
from .estimator import Estimator
from .normalization import Normalization
from .actor_critic import ActorCritic
from .network import (
    HyperEmbedder,
    HyperLERPBlock, 
    HyperPredictHead,
    )


class ActorCriticLatent(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_commands,
        num_critic_commands,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        encoder_hidden_dims=[256, 256, 128],
        prior_hidden_dims=[256, 256, 128],
        embedding_dim=32,
        latent_std=0.0001,
        rnn_type="gru",
        rnn_hidden_dim=256,
        init_std=-1.0,
        const_noise=False,
        **kwargs,
    ):  
        if kwargs:
            print(
                "ActorCriticLatent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_commands=embedding_dim,
            num_critic_commands=num_critic_commands,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            init_std=init_std,
            const_noise=const_noise,
        )
        mlp_input_dim_p = num_actor_obs
        mlp_input_dim_e = num_commands
    
        self.normalizers.update({
            "encoder": Normalization(shape=[mlp_input_dim_e]),
            })
        
        # Posterior Encoder
        num_blocks = len(encoder_hidden_dims)
        encoder_hidden_dim = reduce(math.gcd, encoder_hidden_dims)
        encoder_layers = [
            HyperEmbedder(
                mlp_input_dim_e, encoder_hidden_dim,
                init=math.sqrt(2.0 / encoder_hidden_dim),
                scale=math.sqrt(2.0 / encoder_hidden_dim),
                )
            ]
        num_blocks = len(encoder_hidden_dims)
        for index in range(num_blocks - 1):
            dimensions = math.sqrt(encoder_hidden_dims[index])
            encoder_layers += [
                HyperLERPBlock(
                    encoder_hidden_dim,
                    encoder_hidden_dims[index] // encoder_hidden_dim,
                    scaler_init=math.sqrt(2.0 / encoder_hidden_dim),
                    scaler_scale=math.sqrt(2.0 / encoder_hidden_dim),
                    alpha_init=1.0 / num_blocks,
                    alpha_scale=1.0 / dimensions,
                    )
                ]
        encoder_layers += [
            HyperPredictHead(encoder_hidden_dim, embedding_dim)]
        self.encoder = nn.Sequential(*encoder_layers)

        # Prior Encoder
        self.prior = Estimator(
            num_proprio_obs=mlp_input_dim_p,
            num_estimations=embedding_dim,
            estimator_hidden_dims=prior_hidden_dims,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            )
        
        self.latent_std = latent_std
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def predict_prior(self, observations, hidden_states=None, **kwargs):
        prior_latents = self.prior.predict(observations, hidden_states=hidden_states)
        return torch.tanh(prior_latents)
    
    def predict_prior_inference(self, observations, noise_scale=0.1, hidden_states=None, **kwargs):
        prior_latents = self.prior.predict(observations, hidden_states=hidden_states)
        prior_latents = prior_latents + torch.randn_like(prior_latents) * noise_scale
        return torch.tanh(prior_latents)
    
    def encode(self, commands):
        commands = self.normalizers["encoder"](commands)
        latnets = self.encoder(commands)
        latent_noise = torch.randn_like(latnets) * self.latent_std    
        return torch.tanh(latnets + latent_noise)
    
    def encode_inference(self, commands):
        commands = self.normalizers["encoder"](commands)
        return torch.tanh(self.encoder(commands))
    
    def get_prior_hidden_states(self):
        return self.prior.get_hidden_states()