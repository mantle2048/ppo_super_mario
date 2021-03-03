#!/usr/bin/env python
# coding: utf-8
# %%
import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np

import gym
import utils
import network
from logx import EpochLogger, setup_logger_kwargs

os.environ['CUDA_VISBLE_DEVICES'] = '1'


class PPO:
    def __init__(
        self,
        env,
        actor_critic=network.MLPActorCritic,
        ac_kwargs=dict(),
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        target_kl=0.01,
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
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - 0.01 * dist.entropy().mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        entropy = dist.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data['obs'].to(self.device), data['ret'].to(self.device)
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
    parser.add_argument('--policy', type=str, default='ppo')
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
    parser.add_argument('--target_kl', type=float, default='0.01')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--step_per_epoch', type=int, default=4000)
    parser.add_argument('--max_episode_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--max_timesteps', type=int, default=1e6)
    parser.add_argument('--save_freq', type=int, default=10)
    args = parser.parse_args()

    file_name = f'{args.policy}_{args.env}_{args.seed}'
    print('-----' * 8)
    print(f'Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}')
    print('-----' * 8)

    if not os.path.exists('./results'):
        os.makedirs('./results')

    if not os.path.exists('./models'):
        os.makedirs('./models')


    # Set up logger and save configuration
    logger_kwargs = setup_logger_kwargs(f'{args.policy}_{args.env}', args.seed)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(args)

    # Init Envirorment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)

    kwargs = {
        "env": env,
        "actor_critic": utils.MLPActorCritic,
        "ac_kwargs": dict(hidden_sizes=[args.hidden]*args.layers_len),
        "gamma": args.gamma,
        "clip_ratio": args.clip_ratio,
        "pi_lr": args.pi_lr,
        "vf_lr": args.vf_lr,
        "train_pi_iters": args.train_pi_iters,
        "train_v_iters": args.train_v_iters,
        "lam": args.lam,
        "target_kl": args.target_kl,
        "device": args.device
    }

    policy = PPO(**kwargs)

    # Count variables
    var_counts = tuple(utils.count_vars(module) for module in [policy.ac.pi, policy.ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up model saving
    logger.setup_pytorch_saver(policy.ac.state_dict())

    num_procs = 1
    local_steps_per_epoch = int(args.step_per_epoch / num_procs)
    buf = utils.PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, args.gamma, args.lam)

    # Evaluate untrained policy
    evalutions = [eval_policy(policy, args.env, args.seed)]

    # Prepare for interaction with environment
    start_time = time.time()
    obs, done = env.reset(), False
    episode_ret = 0.
    episode_len = 0.

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(args.epochs):
        for t in range(local_steps_per_epoch):
            act, val, logp = policy.step(obs)

            next_obs, ret, done, _ = env.step(act)
            episode_ret += ret
            episode_len += 1

            # save and log
            buf.add(obs, act, ret, val, logp)
            logger.store(VVals=val)

            # Update obs (critical!)
            obs = next_obs

            timeout = episode_len == args.max_episode_len
            terminal = done or timeout
            epoch_ended = t == local_steps_per_epoch - 1
            if epoch_ended or terminal:
                if epoch_ended and not terminal:
                    print(f'Warning: Trajectory cut off by epoch at {episode_len} steps', flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, val, _ = policy.step(obs)
                else:
                    val = 0
                buf.finish_path(val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=episode_ret, EpLen=episode_len)
                obs, episode_ret, episode_len = env.reset(), 0, 0

        policy.update(buf)

        # Evaluate every Epoch
        print(f'Step: {(epoch + 1) * local_steps_per_epoch}')
        evalution = eval_policy(policy, args.env, args.seed)
        evalutions.append(evalution)
        np.save(f"./results/{file_name}", evalutions)

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
        logger.log_tabular('AverageTestEpRet', evalution)
        logger.dump_tabular()


        # Save model
        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            torch.save(policy.ac.state_dict(), f'./models/{file_name}.pth')
            logger.save_state({'env': env}, None)
