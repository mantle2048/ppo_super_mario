#!/usr/bin/env python
# coding: utf-8
# %%
import torch
import numpy as np
import utils


# %%
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

    def sample(self, batch_size = 100):

        indices = np.random.randint(0, self.size, size = batch_size)
        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.not_done[indices]).to(self.device),
        )


# %%
class PPOBuffer:
    def __init__(self, state_dim, action_dim, size, gamma=0.99, lam=0.95, device="cpu"):
        self.obs_buf = np.zeros(utils.combinded_shape(size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combinded_shape(size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def add(self, state, action, reward, value, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = state
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.log_pros[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf, last_val)
        vals = np.append(self.val_buf, last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1] # Td-error
        self.adv_buf = utils.discount_cumsum(deltas, self.gamma * self.lam)

        self.ret_buf = utils.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs = self.obs_buf, act = self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensors(v, dtype=torch.float32) for k, v in data.items()}


# %%
