import gym
import torch
import joblib
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from yanrl import mp_PPO as PPO
from yanrl import mp_PPO2 as PPO2
from yanrl.user_config import DEFAULT_VIDEO_DIR
from yanrl.utils.wrappers import ReshapeReward, SkipObs, Monitor
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



if __name__ == '__main__' and '__file__' in globals():
    env_name = 'SuperMarioBros-1-1-v0'
    # env_name = 'HalfCheetah-v3'
    policy_type = 'cnn'
    # policy_type = 'mlp'
    policy_name = 'mp_ppo'
    exp_name = policy_name + '_' + env_name

    output=f'../data/{exp_name}/{exp_name}_seed_0'
    video_path = '../video/1-1.mp4'

    env = gym.make(env_name)

    if video_path:
        # monitor = Monitor(256, 240, video_path)
        monitor = None
    else:
        monitor = None

    if 'SuperMario' in env_name:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = ReshapeReward(env, monitor=None)
        env = SkipObs(env)
    env = gym.wrappers.Monitor(env, DEFAULT_VIDEO_DIR, force=True, video_callable=lambda episode_id: True)
    model_name = os.path.join(output, 'pytorch_save', 'model.pth')
    # obs_name = os.path.join(output, 'environment_vars.pkl')

    kwargs = {
        'env': env,
        'ac_kwargs': dict(hidden_sizes=[64]*2),
        'device' : 'cpu'
    }

    policy = PPO2(**kwargs) if policy_type == 'cnn' else PPO(**kwargs)
    policy.ac.load_state_dict(torch.load(model_name))
    # obs_normal=joblib.load(obs_name)['obs_normal']

    for ep in range(10):
        obs = env.reset()[None]
        # obs = obs_normal.normalize(obs, update=False)
        while True:
            act = policy.act(obs)

            nx_obs, rew, done, info = env.step(act[0])
            obs = nx_obs[None]
            # obs = obs_normal.normalize(obs, update=False)
            if done:
                break
