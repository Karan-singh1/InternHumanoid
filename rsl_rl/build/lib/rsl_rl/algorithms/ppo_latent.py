#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as torchd
from typing import Union

from rsl_rl.modules import ActorCriticLatent
from rsl_rl.storage import RolloutStorage


class PPO_LATENT:
    actor_critic: ActorCriticLatent

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        value_clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        smoothness_coef=0.01,
        learning_rate=1e-3,
        learning_rate_limit=[1e-5, 1e-3],
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        overlap_loss_coef=1.0,
        commitment_loss_coef=1.0,
        triplet_alpha=0.25,
        triplet_loss_coef=1.0,
        triplet_num_batch_size=512,
        device="cpu",
        **kwargs,
    ):
        if kwargs:
            print(
                "PPO_LATENT.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
            
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.learning_rate_limit = learning_rate_limit

        # components
        self.actor_critic = actor_critic.to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.optimizer.zero_grad(set_to_none=True)
        
        self.storage = None  # initialized later
        self.transition = None
        
        self.overlap_generator = None
        self.random_generator = None
        self.distance_function = None
        
        # Loss function
        self.regression_loss = nn.MSELoss(reduction='none')
        self.smooth_regression_loss = nn.SmoothL1Loss(reduction='none')

        # parameters
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.smoothness_coef = smoothness_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.overlap_loss_coef = overlap_loss_coef
        self.commitment_loss_coef = commitment_loss_coef
        self.triplet_alpha = triplet_alpha
        self.triplet_loss_coef = triplet_loss_coef
        self.triplet_num_batch_size = triplet_num_batch_size

    def init_storage(
        self, 
        num_envs, 
        num_transitions, 
        obs_shape, 
        critic_obs_shape, 
        command_shape,
        critic_command_shape,
        actions_shape, 
        ):
        self.storage = RolloutStorage(
            num_envs=num_envs, 
            num_transitions=num_transitions,
            obs_shape=obs_shape,
            critic_obs_shape=critic_obs_shape,
            command_shape=command_shape,
            critic_command_shape=critic_command_shape,
            actions_shape=actions_shape,
            device=self.device,
        )
        self.transition = self.storage.Transition()

    def load_extra_components(
        self, 
        overlap_generator,
        random_generator, 
        distance_function,
        ):
        self.overlap_generator = overlap_generator
        self.random_generator = random_generator
        self.distance_function = distance_function

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, commands, critic_commands):
        # Compute the actions and values
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        if self.actor_critic.prior.is_recurrent:
            self.transition.prior_hidden_states = self.actor_critic.get_prior_hidden_states()
        
        self.actor_critic.predict_prior(obs)
        latents = self.actor_critic.encode(commands).detach()
        
        self.transition.actions = self.actor_critic.act(obs, latents).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, critic_commands).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.commands = commands
        self.transition.critic_commands = critic_commands
        return self.transition.actions
    
    def process_post_act(
        self, 
        next_obs, 
        next_critic_obs, 
        next_commands, 
        next_critic_commands,
        ):
        self.transition.next_observations = next_obs
        self.transition.next_critic_observations = next_critic_obs
        self.transition.next_commands = next_commands
        self.transition.next_critic_commands = next_critic_commands

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        timeouts = infos["time_outs"][:, None].float().to(self.device)
        bootstrap = self.transition.values * timeouts
        self.transition.rewards += self.gamma * torch.squeeze(bootstrap, dim=1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones.bool())
        self.actor_critic.prior.reset(dones.bool())

    def compute_returns(self, last_critic_obs, last_critic_commands):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_commands).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_kl_divergence = 0
        mean_overlap_loss = 0
        mean_triplet_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        self.train_mode()
        kl_sum_list = []
        for rollout_samples in generator:
            obs_batch, critic_obs_batch, next_obs_batch, next_critic_obs_batch, \
                commands_batch, critic_commands_batch, next_commands_batch, next_critic_commands_batch, \
                cont_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, \
                hidden_a_batch, hidden_c_batch, hidden_p_batch, optimization_flag = rollout_samples
                
            latents_batch = self.actor_critic.encode(commands_batch)            
            self.actor_critic.act(obs_batch, latents_batch, hidden_states=hidden_a_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            with torch.inference_mode():
                old_dist = torchd.Normal(loc=old_mu_batch, scale=old_sigma_batch)
                new_dist = torchd.Normal(loc=mu_batch, scale=sigma_batch)
                kl_divergence = torchd.kl_divergence(old_dist, new_dist).sum(dim=-1)
                kl_mean = torch.mean(kl_divergence)
                
                kl_sum_list.append(kl_mean.item())
                if optimization_flag and self.desired_kl is not None and self.schedule == "adaptive":
                    epoch_kl_mean = sum(kl_sum_list) / self.num_mini_batches
                    min_learning_rate = min(self.learning_rate_limit)
                    max_learning_rate = max(self.learning_rate_limit)
                    
                    if epoch_kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(min_learning_rate, self.learning_rate / 1.5)
                    elif epoch_kl_mean < self.desired_kl * 0.5 and epoch_kl_mean > 0.0:
                        self.learning_rate = min(max_learning_rate, self.learning_rate * 1.5)
                        
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    
                    kl_sum_list.clear()

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze())
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * \
                torch.clip(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            max_surrogate = torch.max(surrogate, surrogate_clipped) 
            surrogate_loss, entropy_loss = max_surrogate.mean(), entropy_batch.mean()
            
            # Policy function loss
            actor_loss = surrogate_loss - self.entropy_coef * entropy_loss

            # Value function loss
            values_batch = self.actor_critic.evaluate(
                critic_obs_batch, critic_commands_batch, hidden_states=hidden_c_batch)
            if self.use_clipped_value_loss:
                clipped_residual = torch.clip(
                    values_batch - target_values_batch, 
                    min=-self.value_clip_param, max=self.value_clip_param)
                value_losses_clipped = self.smooth_regression_loss(
                    target_values_batch + clipped_residual, returns_batch)
                value_losses = self.smooth_regression_loss(values_batch, returns_batch)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = self.smooth_regression_loss(values_batch, returns_batch).mean()
                
            # Prior Estimation loss
            priors_batch = self.actor_critic.predict_prior(obs_batch, hidden_states=hidden_p_batch)
            consistent_loss = torch.mean(torch.norm(latents_batch.detach() - priors_batch, p=2.0, dim=-1))
            commitment_loss = torch.mean(torch.norm(latents_batch - priors_batch.detach(), p=2.0, dim=-1))
            
             # Latent loss
            with torch.inference_mode():
                anchor_samples_batch, anchor_states, motion_ids, motion_time = self.random_generator(self.triplet_num_batch_size)
                overlap_samples_batch = self.overlap_generator(motion_ids, motion_time)
                
                random_samples_batch1, random_states1, _, _ = self.random_generator(self.triplet_num_batch_size)
                random_samples_batch2, random_states2, _, _ = self.random_generator(self.triplet_num_batch_size)
                
                distances1, distances2 = self.distance_function(anchor_states, random_states1, random_states2)
                condition = (distances1 > distances2).view(self.triplet_num_batch_size)
                
            positive_samples_batch = random_samples_batch1.clone().to(self.device)
            negative_samples_batch = random_samples_batch2.clone().to(self.device)
            positive_samples_batch[condition] = random_samples_batch2[condition].to(self.device)
            negative_samples_batch[condition] = random_samples_batch1[condition].to(self.device)
            
            anchor_latents_batch = self.actor_critic.encode(anchor_samples_batch)
            overlap_latents_batch = self.actor_critic.encode(overlap_samples_batch)
            positive_latents_batch = self.actor_critic.encode(positive_samples_batch)
            negative_latents_batch = self.actor_critic.encode(negative_samples_batch)
            
            overlap_loss = torch.mean(
                torch.norm(anchor_latents_batch - overlap_latents_batch, p=2.0, dim=-1))
            triplet_loss = torch.mean(torch.clip(
                torch.norm(anchor_latents_batch - positive_latents_batch, p=2.0, dim=-1) - 
                torch.norm(anchor_latents_batch - negative_latents_batch, p=2.0, dim=-1) + 
                self.triplet_alpha, min=0.0))
            
            # Smoothness loss
            mixing_weight = cont_batch * torch.rand_like(cont_batch)
            obtain_mix_obs = lambda obs, next_obs: obs + mixing_weight * (next_obs - obs)

            mix_obs_batch = obtain_mix_obs(obs_batch, next_obs_batch)
            mix_commands_batch = obtain_mix_obs(commands_batch, next_commands_batch)
            
            mix_latents_batch = self.actor_critic.encode(mix_commands_batch)
            mix_mu_batch = self.actor_critic.act_inference(
                mix_obs_batch, mix_latents_batch, hidden_states=hidden_a_batch)[0]
            
            policy_smoothness_loss = self.regression_loss(mix_mu_batch, mu_batch.detach()).mean()
            smoothness_loss = self.smoothness_coef * policy_smoothness_loss
            
            # Gradient step
            total_loss = actor_loss + self.value_loss_coef * value_loss + \
                smoothness_loss + consistent_loss + \
                self.overlap_loss_coef * overlap_loss + \
                self.triplet_loss_coef * triplet_loss + \
                self.commitment_loss_coef * commitment_loss
            (total_loss / self.num_mini_batches).backward()
            
            if optimization_flag:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimator_loss += consistent_loss.item()
            mean_kl_divergence += kl_mean.item()
            mean_overlap_loss += overlap_loss.item()
            mean_triplet_loss += triplet_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_kl_divergence /= num_updates
        mean_overlap_loss /= num_updates
        mean_triplet_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_kl_divergence, mean_overlap_loss, mean_triplet_loss
