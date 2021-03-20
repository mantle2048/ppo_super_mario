#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from gym.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def cnn(channels, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(channels)-1):
        act = activation if j < len(channels)-2 else output_activation
        layers += [nn.Conv2d(channels[j], channels[j+1], 3, stride=2,padding=1), act()]
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
            act = act.cpu()
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


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cuda:0"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size=100):

        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.not_done[indices]).to(self.device),
        )



class PPOBuffer:
    def __init__(self, state_dim, action_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def add(self, state, action, reward, value, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = state
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]  # Td-error
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs = self.obs_buf, act = self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPO_mp_Buffer:
    def __init__(self, state_dim, action_dim, size, gamma=0.99, lam=0.95, cpu=1):
        self.obs_buf = np.zeros(combined_shape(cpu, ((size,) +  state_dim)), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(cpu, ((size,) +  action_dim)), dtype=np.float32)
        self.rew_buf = np.zeros((cpu, size), dtype=np.float32)
        self.val_buf = np.zeros((cpu, size), dtype=np.float32)
        self.adv_buf = np.zeros((cpu, size), dtype=np.float32)
        self.ret_buf = np.zeros((cpu, size), dtype=np.float32)
        self.logp_buf = np.zeros((cpu, size), dtype=np.float32)

        self.gamma, self.lam, self.cpu = gamma, lam, cpu
        self.ptr, self.path_start_idx, self.max_size = 0, np.zeros(self.cpu,dtype=np.int), size

    def add(self, state, action, reward, value, logp):
        assert self.ptr < self.max_size
        self.obs_buf[:, self.ptr] = state
        self.act_buf[:, self.ptr] = action
        self.rew_buf[:, self.ptr] = reward
        self.val_buf[:, self.ptr] = value
        self.logp_buf[:, self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, proc_idx=0):
        assert proc_idx >= 0 and proc_idx < self.cpu
        path_slice = slice(self.path_start_idx[proc_idx], self.ptr)
        rews = np.append(self.rew_buf[proc_idx, path_slice], last_val)
        vals = np.append(self.val_buf[proc_idx, path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]  # Td-error
        self.adv_buf[proc_idx, path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[proc_idx, path_slice]= discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx[proc_idx] = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, np.zeros(self.cpu, dtype=np.int)

        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs = self.obs_buf, act = self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).view(self.max_size * self.cpu, -1).squeeze() \
                for k, v in data.items()}

class dropped_ObsNormalize():
    def __init__(self, shape, clip=5.0):
        self.n = 0
        self.mean = np.zeros(shape)
        self.mean_diff = np.zeros(shape)
        self.var = np.zeros(shape)
        self.clip = clip

    def add(self, obs):
        '''
        by math
        En = En-1 + (xn - En-1) / n
        Fn = Fn-1 + (xn - En-1) * (xn - En)
            Fn = n * Var_n | Fn-1 = (n - 1) * Var_n-1
        So Var_n = (n-1)/n * (xn - En-1)**2 + (n-1)/n * Var_n-1
        '''
        assert obs.shape == self.mean.shape, "obs must be the same dim"
        self.n += 1
        old_mean = self.mean.copy()
        self.mean += (obs - self.mean) / self.n
        self.mean_diff += (obs - old_mean) * (obs - self.mean)
        self.var = self.mean_diff/self.n if self.n > 1 else np.square(self.mean)

    def normalize(self, obs):
        obs = np.asarray(obs)
        self.add(obs)
        obs =  (obs - self.mean) / np.sqrt(self.var)
        return np.clip(obs, -self.clip, self.clip)

    def normalize_without_add(self, obs, idx):
        obs = np.asarray(obs)
        obs =  (obs - self.mean[idx]) / np.sqrt(self.var[idx])
        return np.clip(obs, -self.clip, self.clip)



class ObsNormalize():
    def __init__(self, shape, cpu=1, clip=5.0):
        self.cpu = cpu
        assert self.cpu >= 1
        self.n = 0
        self.mean = np.zeros(shape)
        self.mean_diff = np.zeros(shape)
        self.var = np.zeros(shape)
        self.clip = clip

    def add(self, obs):
        '''
        by math
        En = En-1 + (xn - En-1) / n
        Fn = Fn-1 + (xn - En-1) * (xn - En)
            Fn = n * Var_n | Fn-1 = (n - 1) * Var_n-1
        So Var_n = (n-1)/n * (xn - En-1)**2 + (n-1)/n * Var_n-1
        '''
        self.n += 1
        if self.n == 1:
            self.mean = obs
        else:
            old_mean = self.mean.copy()
            self.mean += (obs - self.mean) / self.n
            self.mean_diff += (obs - old_mean) * (obs - self.mean)
            self.var = self.mean_diff/self.n if self.n > 1 else np.square(self.mean)

    def normalize(self, obs, update=True):
        ''' for single obs '''
        assert obs.shape == self.mean.shape, "obs must be the same dim"
        obs = np.asarray(obs)
        if update:
            self.add(obs)
        obs =  (obs - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(obs, -self.clip, self.clip)

    def normalize_all(self, obs, update=True):
        ''' for multi obss '''
        assert (obs.shape[-1],) == self.mean.shape, "obs must be the same dim"
        obs = np.asarray(obs)
        return np.asarray([self.normalize(obs[idx]) for idx in range(self.cpu)])
