import os
import time
import argparse
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gym
import gym_super_mario_bros
from yanrl import PPO as PPO
from yanrl import PPO2 as PPO2
from yanrl import Logger, EpochLogger
from yanrl.utils.env import make_envs
from yanrl.utils.logx import setup_logger_kwargs
import yanrl.utils.core as core
from yanrl.user_config import DEFAULT_MODEL_DIR


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='ppo')
    parser.add_argument('--policy_type', type=str, default='cnn')
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
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--max_episode_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--max_timesteps', type=int, default=1e6)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--obs_norm',action='store_true')
    parser.add_argument('--obs_clip',type=float,default=5.0)
    parser.add_argument('--use_clipped_value_loss',action='store_true')
    parser.add_argument('--clip_val_param',type=float, default=80.0)
    args = parser.parse_args()

    if args.cpu > 1: args.policy = 'mp_' + args.policy

    file_name = f'{args.policy}_{args.env}_{args.seed}'
    print('-----' * 8)
    print(f'Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}')
    print('-----' * 8)


    if not os.path.exists(DEFAULT_MODEL_DIR):
        os.makedirs(DEFAULT_MODEL_DIR)

    # Set up logger and save configuration
    logger_kwargs = setup_logger_kwargs(f'{args.policy}_{args.env}', args.seed, datestamp=args.datestamp)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(args)

    # Init Envirorment
    env = make_envs(args.env, args.cpu, args.seed) # SingleEnv Wrapper and env.seed env.action_space seed
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape if  isinstance(env.action_space, gym.spaces.Box) else (1, )

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = {
        "env": env,
        "actor_critic": core.CNNActorCritic if args.policy_type == 'cnn' else core.MLPActorCritic,
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
        "device": args.device,
        "logger": logger
    }

    policy = PPO2(**kwargs) if args.policy_type == 'cnn' else PPO(**kwargs)
    # policy.ac.share_memory()

    # Count variables
    if args.policy_type == 'mlp':
        var_counts = tuple(core.count_vars(module) for module in [policy.ac.pi, policy.ac.v])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
    else:
        var_counts = core.count_vars(policy.ac)
        logger.log('\nNumber of parameters: \t pi_v: %d\n'%var_counts)


    # Set up model saving
    logger.setup_pytorch_saver(policy.ac.state_dict())

    local_steps_per_epoch = int(args.steps_per_epoch / args.cpu)
    if args.cpu == 0:
        buf = core.PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, args.gamma, args.lam)
    else:
        buf = core.PPO_mp_Buffer(obs_dim, act_dim, local_steps_per_epoch, args.gamma, args.lam, args.cpu)
    # Prepare for interaction with environment
    start_time = time.time()
    obs, done = env.reset(), [False for _ in range(args.cpu)]
    if args.obs_norm:
        ObsNormal = core.ObsNormalize(obs_dim, args.cpu, args.obs_clip) # Normalize the observation
        obs = ObsNormal.normalize_all(obs)
    episode_ret = np.zeros(args.cpu, dtype=np.float32)
    episode_len = np.zeros(args.cpu)


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(args.epochs):
        for t in range(local_steps_per_epoch):
            act, val, logp = policy.step(obs)

            next_obs, ret, done, info = env.step(act)
            if args.obs_norm:
                next_obs = ObsNormal.normalize_all(next_obs)
            episode_ret += ret
            episode_len += 1

            # save and log
            buf.add(obs, act, ret, val, logp)
            logger.store(VVals=val)

            # Update obs (critical!)
            obs = next_obs
            # In multiprocess env when a episode is terminal it will automatic reset(This has been removed because hard to reset())
            # the next_obs is the obs after reset,the real obs that cause terminal is stored in info['terminal_observation'] | updata: automatic reset has been removed

            timeout = episode_len == args.max_episode_len
            terminal = done + timeout
            epoch_ended = t == local_steps_per_epoch - 1


            # 感觉写的太臃肿了，暂时没想到好的写法

            for idx in range(args.cpu):
                if epoch_ended or terminal[idx]:
                    if epoch_ended and not terminal[idx]:
                        print(f'Warning: Trajectory {idx} cut off by epoch at {episode_len[idx]} steps', flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout[idx] or epoch_ended:
                        _, val, _ = policy.step(obs[idx][None])
                    else:
                        val = 0
                    buf.finish_path(val, idx)
                    if terminal[idx]:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=episode_ret[idx], EpLen=episode_len[idx])
                    obs[idx], episode_ret[idx], episode_len[idx] = env.reset(idx), 0, 0
                    # if args.obs_norm: obs = ObsNormal.normalize(obs)

        # During Experiment, I find that reset state without Normalize will perform better

        policy.update(buf)

        # Log info about epoch
        logger.log_tabular('Exp', file_name)
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*args.steps_per_epoch)
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
            torch.save(policy.ac.state_dict(), f'{DEFAULT_MODEL_DIR}/{file_name}.pth')
            logger.save_state(dict(obs_normal=ObsNormal if args.obs_norm else None), None)

if __name__ == '__main__':
    train()
