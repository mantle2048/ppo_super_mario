import gym
import torch
import joblib
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from yanrl import PPO as PPO
from yanrl import PPO2 as PPO2
from yanrl.user_config import DEFAULT_VIDEO_DIR
from yanrl.utils.wrappers import ReshapeReward, SkipObs, Monitor, SingleEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



if __name__ == '__main__' and '__file__' in globals():
    env_name = 'SuperMarioBros-1-2-v0'
    # env_name = 'HalfCheetah-v3'
    policy_type = 'cnn'
    # policy_type = 'mlp'
    policy_name = 'mp_ppo'
    exp_name = policy_name + '_' + env_name

    output=f'../data/{exp_name}/{exp_name}_seed_0'

    env = gym.make(env_name)

    if 'SuperMario' in env_name:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = ReshapeReward(env, monitor=None)
        env = SkipObs(env)
    env = SingleEnv(env)
    env = gym.wrappers.Monitor(env, DEFAULT_VIDEO_DIR, force=True, video_callable=lambda episode_id: True)
    model_name = os.path.join(output, 'pytorch_save', 'model.pth')
    obs_name = os.path.join(output, 'environment_vars.pkl')

    kwargs = {
        'env': env,
        'ac_kwargs': dict(hidden_sizes=[64]*2),
        'device' : 'cpu'
    }

    policy = PPO2(**kwargs) if policy_type == 'cnn' else PPO(**kwargs)
    policy.ac.load_state_dict(torch.load(model_name, map_location='cpu'))
    obs_normal=joblib.load(obs_name)['obs_normal']
    obs_normal.cpu=1
    # This Command for slower(setpts > 1.0) of faster(setpts < 1.0) video
    # ffmpeg -r 60 -i input.mp4 -filter:v "setpts=2.0*PTS" output.mp4
    # This Command for add audio to video(Magic)
    # ffmpeg -i video.mp4 -i audio.mp3 -map 0:v -map 1:a -codec copy -shortest out.mp4
    # Source: https://stackoverflow.com/questions/20254846/how-to-add-an-external-audio-track-to-a-video-file-using-vlc-or-ffmpeg-command-l
    for ep in range(10):
        obs = env.reset()
        obs = obs_normal.normalize_all(obs, update=False)
        while True:
            act = policy.act(obs)
            nx_obs, rew, done, info = env.step(act)
            obs = nx_obs
            obs = obs_normal.normalize_all(obs, update=False)
            if done:
                break
