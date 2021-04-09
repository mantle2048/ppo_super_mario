import gym
import torch
import numpy as np
from gym import Wrapper
from gym.spaces import Box
import cv2
import subprocess as sp

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

def process_frame(obs):
    if obs is not None:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84,84))[None, :, :] / 255.
        return obs
    else:
        return np.zeros((1, 84, 84))


class SingleEnv(gym.Wrapper):
    ''' Wrapper gym.make(env_id) env for obs, rew, done unsqueeze in the first dim'''
    def __init__(self, env=None):
        gym.Wrapper.__init__(self, env)

    def reset(self, idx=None):
        ''' idx param just for api consistent'''
        return self.env.reset()[None]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs[None], np.array([rew]), np.array([done]), info


class ReshapeReward(gym.Wrapper):
    def __init__(self, env=None, monitor=None):
        gym.Wrapper.__init__(self, env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84,84))
        self.cur_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(obs)
        obs = process_frame(obs)
        rew += (info["score"] - self.cur_score) / 40.
        self.cur_score = info["score"]
        if done:
            if info["flag_get"]:
                rew += 100.
            else:
                rew -= 100.
        return obs, rew / 10., done, info

    def reset(self):
        self.cur_score = 0
        return process_frame(self.env.reset())

class SkipObs(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self.observation_space = Box(low=0, high=255, shape=(skip,84,84))
        self.skip = skip
        self.obss = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_rew = 0.
        last_obss = []
        for i in range(self.skip):
            obs, rew, done, info = self.env.step(action)
            total_rew += rew
            if i >= self.skip / 2:
                last_obss.append(obs)
            if done:
               return self.obss, total_rew, done, info
        max_obs = np.max(np.concatenate(last_obss, 0), 0)
        self.obss[:-1] = self.obss[1:]
        self.obss[-1] = max_obs
        return self.obss, total_rew, done, info

    def reset(self):
        obs = self.env.reset()
        self.obss = np.concatenate([obs for _ in range(self.skip)], 0)
        return self.obss
