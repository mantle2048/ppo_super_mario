import gym
import torch
import joblib
import os
import yanrl.mp_PPO import PPO

env = gym.make('HalfCheetah-v3')
env = gym.wrappers.Monitor(env, 'video', force=True)

output='./data/mp_ppo_HalfCheetah-v3/mp_ppo_HalfCheetah-v3_seed_0'
model_name = os.path.join(output, 'pytorch_save', 'model.pth')
obs_name = os.path.join(output, 'environment_vars.pkl')
kwargs = {
    'env': env,
    'ac_kwargs': dict(hidden_sizes=[64]*2),
    'device' : 'cpu'
}

policy = PPO(**kwargs)
policy.ac.load_state_dict(torch.load(model_name))
obs_normal=joblib.load(obs_name)['obs_normal']

for ep in range(1):
    obs = env.reset()
    obs = obs_normal.normalize(obs, update=False)
    while True:
        env.render()
        act = policy.act(obs)
        nx_obs, rew, done, info = env.step(act)
        obs = nx_obs
        obs = obs_normal.normalize(obs, update=False)
        if done:
            break
