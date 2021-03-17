#!/usr/bin/env python
# coding: utf-8
# %%
import os
import time
import argparse
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

import gym
import core.ppo_core as core
from utils.logx import EpochLogger, setup_logger_kwargs
from utils.env import make_mp_envs

os.environ['CUDA_VISBLE_DEVICES'] = '1'


class PPO:
    def __init__(
        self,
        env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        target_kl=0.01,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        clip_val_param = 80.0,
        device='cuda'
    ):
        # Create actor-critic module
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.target_kl = target_kl
        self.max_grad_norm=max_grad_norm
        self.use_clipped_value_loss=use_clipped_value_loss
        self.clip_val_param =  clip_val_param  # This value depends on environment and this can decrease the var of value functino
        self.device = device

        self.ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(self.device)
        self.pi_optim = torch.optim.Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.v_optim = torch.optim.Adam(self.ac.v.parameters(), lr=self.vf_lr)


    def step(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        return self.ac._step(obs)

    def act(self, obs):
        return self.step(obs)[0]

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'].to(self.device), \
                                  data['act'].to(self.device), \
                                  data['adv'],                 \
                                  data['logp']
        # Policy loss
        dist, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - 0.01 * dist.entropy().mean()
        # whether to use entropy depends on the performence

        # Useful extra info
        approx_kl = (logp_old - logp).mean().abs().item()
        entropy = dist.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data['obs'].to(self.device), data['ret'].to(self.device)
        if self.use_clipped_value_loss:
            val_old = data['val'].to(self.device)
            val = self.ac.v(obs)
            val_clipped = val_old + (val - val_old).clamp(-self.clip_val_param, self.clip_val_param)
            v_loss = (val - ret).pow(2)
            v_loss_clipped = (val_clipped - ret).pow(2)
            v_loss = torch.max(v_loss, v_loss_clipped).mean()
            return v_loss
        else:
            return F.mse_loss(self.ac.v(obs), ret)

    def update(self, buf):
        data = buf.get()

        pi_loss_old, pi_info_old = self.compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = self.compute_loss_v(data)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optim.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                logger.log(f'Early stopping at step {i} due to reaching max kl.')
                break
            loss_pi.backward()
            self.pi_optim.step()
        logger.store(StopIter=i)

        for i in range(self.train_v_iters):
            self.v_optim.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.v_optim.step()

        # Log changes from update
        kl, entropy, clipfrac = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(
            LossPi=pi_loss_old,
            LossV=v_loss_old,
            KL=kl,
            Entropy=entropy,
            ClipFrac=clipfrac,
            DeltaLossPi=(loss_pi.item() - pi_loss_old),
            DeltaLossV=(loss_v.item() - v_loss_old)
        )


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# Only off-policy algos need this
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            act = policy.act(obs)
            obs, ret, done, _ = eval_env.step(act)
            avg_reward += ret
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='mp_ppo')
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers_len', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--train_pi_iters', type=int,default=80)
    parser.add_argument('--train_v_iters', type=int,default=80)
    parser.add_argument('--lam', type=float,default=0.97)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--datestamp', action='store_true')
    parser.add_argument('--step_per_epoch', type=int, default=4000)
    parser.add_argument('--max_episode_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--max_timesteps', type=int, default=1e6)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--obs_norm',action='store_true')
    parser.add_argument('--obs_clip',type=float,default=5.0)
    parser.add_argument('--use_clipped_value_loss',action='store_true')
    parser.add_argument('--clip_val_param',type=float, default=80.0)
    args = parser.parse_args()

    file_name = f'{args.policy}_{args.env}_{args.seed}'
    print('-----' * 8)
    print(f'Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}')
    print('-----' * 8)


    if not os.path.exists('./models'):
        os.makedirs('./models')


    # Set up logger and save configuration
    logger_kwargs = setup_logger_kwargs(f'{args.policy}_{args.env}', args.seed, datestamp=args.datestamp)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(args)

    # Init Envirorment
    env = make_mp_envs(args.env, args.cpu, args.seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = {
        "env": env,
        "actor_critic": core.MLPActorCritic,
        "ac_kwargs": dict(hidden_sizes=[args.hidden]*args.layers_len),
        "gamma": args.gamma,
        "clip_ratio": args.clip_ratio,
        "pi_lr": args.pi_lr,
        "vf_lr": args.vf_lr,
        "train_pi_iters": args.train_pi_iters,
        "train_v_iters": args.train_v_iters,
        "lam": args.lam,
        "target_kl": args.target_kl,
        "use_clipped_value_loss": args.use_clipped_value_loss,
        "clip_val_param": args.clip_val_param,
        "device": args.device
    }

    policy = PPO(**kwargs)
    policy.ac.share_memory()

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [policy.ac.pi, policy.ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up model saving
    logger.setup_pytorch_saver(policy.ac.state_dict())

    local_steps_per_epoch = int(args.step_per_epoch / args.cpu)
    buf = core.PPO_mp_Buffer(
        obs_dim,
        act_dim,
        local_steps_per_epoch,
        args.gamma,
        args.lam,
        args.cpu
    )
    # Prepare for interaction with environment
    start_time = time.time()
    obs, done = env.reset(), [False for _ in range(args.cpu)]
    if args.obs_norm:
        ObsNormal = core.mp_ObsNormalize(obs_dim, args.cpu, args.obs_clip) # Normalize the observation
        obs = ObsNormal.normalize_all(obs)
    episode_ret = np.zeros(args.cpu, dtype=np.float32)
    episode_len = np.zeros(args.cpu)


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(args.epochs):
        for t in range(local_steps_per_epoch):
            act, val, logp = policy.step(obs)

            next_obs, ret, done, info = env.step(act)
            episode_ret += ret
            episode_len += 1

            # save and log
            buf.add(obs, act, ret, val, logp)
            logger.store(VVals=val)

            # Update obs (critical!)
            obs = next_obs
            # In multiprocess env when a episode is terminal it will automatic reset and 
            # the next_obs is the obs after reset,the real obs that cause terminal is stored in info['terminal_observation']

            timeout = episode_len == args.max_episode_len
            terminal = done + timeout
            epoch_ended = t == local_steps_per_epoch - 1


            # 感觉写的太臃肿了，暂时没想到好的写法
            for idx in range(args.cpu):
                # if not terminal[idx] and not epoch_ended:
                #     continue

                if terminal[idx]:
                    if timeout[idx]:
                         ter_obs = info[idx]['terminal_observation']
                         ter_obs = ObsNormal.normalize(ter_obs)
                         _, val, _ = policy.step(ter_obs)
                    else:
                        val = 0
                    buf.finish_path(val, idx)
                    logger.store(EpRet=episode_ret[idx], EpLen=episode_len[idx])

                elif epoch_ended:
                    print(f'Warning: Trajectory {idx} cut off by epoch at {episode_len[idx]} steps', flush=True)
                    ter_obs = obs[idx]
                    ter_obs = ObsNormal.normalize(ter_obs)
                    _, val, _ = policy.step(ter_obs)
                    buf.finish_path(val, idx)
                else:
                     if args.obs_norm:
                         obs[idx] = ObsNormal.normalize(obs[idx])
                         continue
                episode_ret[idx], episode_len[idx] =  0., 0

            # for idx in range(args.cpu):
            #     if args.obs_norm and not terminal[idx]:
            #         obs[idx] = ObsNormal.normalize(obs[idx])

        obs = env.reset()
        # obs = ObsNormal.normalize_all(obs)

        policy.update(buf)

        # Log info about epoch
        logger.log_tabular('Exp', file_name)
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*args.step_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', int((time.time()-start_time) / 60))
        if args.obs_norm:
            logger.log_tabular('obs_mean', ObsNormal.mean.mean())
            logger.log_tabular('obs_std', np.sqrt(ObsNormal.var).mean())
        logger.dump_tabular()


        # Save model
        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            torch.save(policy.ac.state_dict(), f'./models/{file_name}.pth')
            logger.save_state({'env': env}, None)
            # I don't know how to save multi processing env
