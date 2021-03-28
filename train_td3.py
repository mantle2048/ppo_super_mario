import os
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
from yanrl import TD3
from yanrl import EpochLogger
from yanrl.utils.env import make_envs
from yanrl.utils.logx import setup_logger_kwargs
import yanrl.utils.core as core
from yanrl.user_config import DEFAULT_MODEL_DIR


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='td3')
    parser.add_argument('--policy_type', type=str, default='mlp')
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layers_len', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--start_timesteps', type=int, default=int(20e3))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--max_episode_len', type=int, default=1000)
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--q_lr', type=float, default=3e-4)
    parser.add_argument('--expl_noise', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--policy_freq', type=int, default=2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--datestamp', action='store_true')
    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--obs_clip', type=float, default=5.0)
    args = parser.parse_args()

    if args.cpu > 1:
        args.policy = 'mp_' + args.policy

    file_name = f'{args.policy}_{args.env}_{args.seed}'
    print('-----' * 8)
    print(f'Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}')
    print('-----' * 8)

    if not os.path.exists(DEFAULT_MODEL_DIR):
        os.makedirs(DEFAULT_MODEL_DIR)

    # Setup logger and save config
    logger_kwargs = setup_logger_kwargs(
        f'{args.policy}_{args.env}',
        args.seed,
        datestamp=args.datestamp
    )
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(args)

    # Init Envirorment
    env = make_envs(args.env, args.cpu, args.seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Set Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    kwargs = {
        'env': env,
        'actor_critic': core.MLPDetActorCritic,
        'ac_kwargs': dict(hidden_sizes=[args.hidden_size]*args.layers_len),
        'gamma': args.gamma,
        'batch_size': args.batch_size,
        'tau': args.tau,
        'expl_noise': args.expl_noise,
        'policy_noise': args.policy_noise,
        'policy_freq': args.policy_freq,
        'pi_lr': args.pi_lr,
        'q_lr': args.q_lr,
        'noise_clip': args.noise_clip,
        'device': args.device,
        'logger': logger
    }

    policy = TD3(**kwargs) if args.policy_type == 'mlp' else TD3(**kwargs)

    # Count variables
    if args.policy_type == 'mlp':
        var_counts = tuple(core.count_vars(module) for module in [policy.ac.pi, policy.ac.q1, policy.ac.q2])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d,  \t q2: %d\n'%var_counts)
    else:
        var_counts = core.count_vars(policy.ac)
        logger.log('\nNumber of parameters: \t  pi_q1_q2: %d\n'%var_counts)

    # Set up model saving
    logger.setup_pytorch_saver(policy.ac.state_dict())

    buf = core.ReplayBuffer(obs_dim, act_dim, size=args.buffer_size, device=args.device)

    # Prepare for interaction with environment
    total_steps = args.steps_per_epoch * args.epochs
    start_time = time.time()
    obs, done = env.reset(), [False for _ in range(args.cpu)]
    if args.obs_norm:
        ObsNormal = core.ObsNormalize(obs_dim, args.cpu, args.obs_clip)  # Normalize the observation
        obs = ObsNormal.normalize_all(obs)
    episode_rew = np.zeros(args.cpu, dtype=np.float32)
    episode_len = np.zeros(args.cpu)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(0, total_steps, args.cpu):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Aferwards,
        # use the learned policy (with some noise, via act_noise)
        if t > args.start_timesteps:
            act = policy.select_action(obs)
        else:
            act = env.action_space.sample()

        # Step the env
        next_obs, rew, done, _ = env.step(act)
        if args.obs_norm:
            next_obs = ObsNormal.normalize_all(next_obs)
        episode_rew += rew
        episode_len += 1

        # Ignore the done "done" signal if it comes from hitting the time
        # horizon (that is , when it's an artificial terinal signal
        # that isn't based on the agent's state
        for idx, d in enumerate(done):
            done[idx] = False if episode_len[idx] == args.max_episode_len else d

        # Store experience to repaly buffer
        buf.add(obs, act, rew, next_obs, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        obs = next_obs

        for idx in range(args.cpu):
            if done[idx] or (episode_len[idx] == args.max_episode_len):
                logger.store(EpRet=episode_rew[idx], EpLen=episode_len[idx])
                obs, episode_rew[idx], episode_len[idx] = env.reset(idx), 0, 0

        # Update handling
        # Update every timesteps or Update 50 times each 50 timesteps
        if t > args.start_timesteps:
            batch = buf.sample(args.batch_size)
            policy.update(batch)

        # End of epoch handling
        if t >= args.start_timesteps and (t+1) % args.steps_per_epoch == 0:
            epoch = (t+1) // args.steps_per_epoch

            # Test the performance of the deterministic version of the agent
            policy.eval_policy(args.env, args.seed)

            # Log info about epoch
            logger.log_tabular('Exp', file_name)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEplen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t+1)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', (time.time() - start_time) / 60)
            if args.obs_norm:
                logger.log_tabular('obs_mean', ObsNormal.mean.mean())
                logger.log_tabular('obs_std', np.sqrt(ObsNormal.var).mean())
            logger.dump_tabular()

            # Save model
            if (epoch % args.save_freq == 0) or (epoch == args.epochs):
                logger.save_state(dict(obs_normal=ObsNormal if args.obs_norm else None), None)


if __name__ == '__main__':
    train()
