#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from gym.spaces import Box, Discrete
import numpy as np


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        Actor.__init__(self)
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs).cpu()
        return Categorical(logits=logits)  # note the logits not the probs | and logits are numbers before softmax

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)



class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        Actor.__init__(self)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.register_parameter('log_std', self.log_std)

    def _distribution(self, obs):
        mu = self.mu_net(obs).cpu()
        std = torch.exp(self.log_std).cpu()  # see std as log_std and do exp to prevent negative std
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # do sum with the dimonsion of log_prob and then exp the sum you can konw that the multi probs of action
        return pi.log_prob(act).sum(dim=-1)  # Last axis sum needed for Torch Normal distribution



class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), dim=-1)  # Critical to ensure v has right shape



class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes, activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)

        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value functino
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def _step(self, obs):
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            a = dist.sample()
            logp_a = self.pi._log_prob_from_distribution(dist, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def forward(self):
        raise NotImplementedError
