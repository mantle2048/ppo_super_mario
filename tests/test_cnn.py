import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import torch
import torch.nn as nn
import numpy as np
import gym_super_mario_bros as gym
from yanrl.utils.core import CNNActorCritic, MLPActorCritic
if __name__ == '__main__':
    env = gym.make('SuperMarioBros-1-1-v0')
    cnn_ac = CNNActorCritic(env.observation_space, env.action_space)
    mlp_ac = MLPActorCritic(env.observation_space, env.action_space, [256, 64, 64])
    obs = torch.rand(10, *env.observation_space.shape).permute(0, 3, 1, 2)
    obs = torch.rand(10, 3, 240, 256)
    act = torch.as_tensor([env.action_space.sample() for _ in range(10)])
    import ipdb; ipdb.set_trace()
    pi = cnn_ac.pi(obs, act)
    v = cnn_ac.v(obs)
    a, v, logp_a = cnn_ac._step(obs)
