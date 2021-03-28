from copy import deepcopy
import itertools
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yanrl.utils.core as core
from yanrl.utils.logx import EpochLogger
from yanrl.utils.wrappers import SingleEnv

class TD3:

    def __init__(
        self,
        env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        gamma=0.99,
        batch_size=256,
        tau=0.005,
        expl_noise=0.1,
        policy_noise=0.2,
        policy_freq=2,
        pi_lr=3e-4,
        q_lr=3e-4,
        noise_clip=0.5,
        device='cpu',
        logger=None
    ):
        self.obs_dim = env.observation_space
        self.act_dim = env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.max_action = env.action_space.high[0]
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.noise_clip = noise_clip
        self.device = device

        # Create actor-critic module and target networks
        self.ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(self.device)
        self.ac_target = deepcopy(self.ac)

        # Freeze target networks with respet to optimizers (only updata via tau averaging)
        for p in self.ac_target.parameters():
            p.require_grad = False

        # List of parameters for both Q-networks(save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Optimizers for policy and q-function
        self.pi_optim = torch.optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optim = torch.optim.Adam(self.q_params, lr=q_lr)

        self.logger = logger
        self.total_it = 0

    def compute_loss_q(self, data):
        ''' Function for computing TD3 Q-losses '''
        obs, act, rew, next_obs, done = data['obs'], data['act'], data['rew'], data['next_obs'], data['done']

        # Bellman backup for Q functions
        with torch.no_grad():

            # Target policy smoothing
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.ac_target.pi(next_obs) + noise).clamp(-self.max_action, self.max_action)

            # Target Q-values
            next_q1_target = self.ac_target.q1(next_obs, next_act)
            next_q2_target = self.ac_target.q2(next_obs, next_act)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            q_target = rew + self.gamma * (1 - done) * next_q_target

        # Mse Loss against Bellman backup
        q1 = self.ac.q1(obs, act)
        q2 = self.ac.q2(obs, act)
        loss_q = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Useful info for logging
        loss_info = dict(
            Q1Vals=q1.cpu().detach().numpy(),
            Q2Vals=q2.cpu().detach().numpy()
        )

        return loss_q, loss_info

    def compute_loss_pi(self, data):
        ''' Function for computing TD3 pi loss '''
        obs = data['obs']
        q1_pi = self.ac.q1(obs, self.ac.pi(obs))
        return -q1_pi.mean()

    def update(self, data):

        self.total_it += 1
        self.q_optim.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optim.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **loss_info)

        # Possible update pi and target network
        if self.total_it % self.policy_freq == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi
            self.pi_optim.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optim.step()

            # Unfreeze Q-networks so you can optimize if at next step.
            for p in self.q_params:
                p.requires_grad = True

            # Record things
            self.logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by tau averaging.
            with torch.no_grad():
                for p, p_target in zip(self.ac.parameters(), self.ac_target.parameters()):
                    # Note: in-place operations 'mul_', 'add_' to update target params
                    p_target.data.mul_(1 - self.tau)
                    p_target.data.add_(self.tau * p.data)

    def select_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        act = self.ac.act(obs)  # Return np.nparray fmt
        act += np.random.normal(0, self.max_action*self.expl_noise, size=self.act_dim)
        return np.clip(act, -self.max_action, self.max_action)

    def eval_policy(self, env_name, seed, eval_episodes=5):

        '''Runs policy for X episodes and returns average reward
        A fixed seed is used for the eval environment
        Only off-policy algos need this'''

        eval_env = gym.make(env_name)
        eval_env = SingleEnv(eval_env)
        eval_env.seed(seed+100)
        # Assert env is wrapped by TimeLimit and env has attr _max_episode_steps
        for _ in range(eval_episodes):
            obs, done, ep_rew, ep_len = eval_env.reset(), False, 0., 0
            while not done:
                # Take deterministic actions at test time (noise=0)
                obs, rew, done, _ = eval_env.step(self.select_action(obs))
                ep_rew += rew
                ep_len += 1
            self.logger.store(TestEpRet=ep_rew, TestEplen=ep_len)
