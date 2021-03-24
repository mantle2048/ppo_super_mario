#! /usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import torch.multiprocessing as mp
import gym
from gym.spaces import Box, Discrete

import pickle
import cloudpickle

import gym_super_mario_bros
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from yanrl.utils.wrappers import ReshapeReward, SkipObs
from yanrl.utils.env import make_mp_envs

if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    env = make_mp_envs('SuperMarioBros-1-1-v0', 3, 0)

