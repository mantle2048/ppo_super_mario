#!/usr/bin/env python
import numpy as np
import torch.multiprocessing as mp
import gym
from gym.spaces import Box, Discrete

import pickle
import cloudpickle

class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)

class AlreadySteppingError(Exception):
    """
    Raised when as asynchronout step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        """Return state values to be pickled."""
        return cloudpickle.dumps(self.x)

    def __setstate__(self, obs):
        """Restore state from the unpickled state values."""
        self.x = pickle.loads(obs)

    def __call__(self):
        return self.x()


class MultiEnv:
    def __init__(self, env_id, num_env):
        assert num_env > 0, 'num_env must be postive'
        self.num_env = num_env
        self.envs = []
        for _ in range(num_env):
            self.envs.append(gym.make(env_id))
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self): # MultiEnv reset
        for env in self.envs:
            env.reset()

    def step(self, actions): # MultiEnv step
        assert self.num_env == len(actions)
        nx_obs = []
        rets = []
        dones = []
        infos = []

        for env, act in zip(self.envs, actions):
            nx_ob, ret, done, info = env.step(act)
            nx_obs.append(nx_ob)
            rets.append(ret)
            dones.append(done)
            infos.append(info)

            if done:
                env.reset()

        return np.asarray(nx_obs), np.asarray(rets), \
                np.asassay(dones), np.asarray(infos)

    def sample(self): # MultiEnv sample
         acts = []
         for _ in range(self.num_env):
             acts.append(self.envs[0].action_space.sample())
         return acts


def worker(env_conn, agent_conn, env_fn):
    agent_conn.close()
    env = env_fn()
    while True:
        cmd, action = env_conn.recv()
        if cmd == 'step':
            nx_obs, ret, done, info = env.step(action)
            if done:
                info['terminal_observation'] = nx_obs
                nx_obs = env.reset()
            env_conn.send((nx_obs, ret, done, info))

        elif cmd == 'reset':
            obs = env.reset()
            env_conn.send((obs, ))

        elif cmd == 'render':
            env_conn.send(env.render())

        elif cmd == 'sample':
            nx_obs, ret, done, info = env.step(env.action_space.sample())
            if done:
                nx_obs = env.reset()
            env_conn.send((nx_obs, ret, done, info))

        elif cmd == 'close':
            env_conn.close()
            break
        else:
            raise NotImplementedError

class SubprocVecEnv:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        num_of_envs = len(env_fns)
        tmp_env = env_fns[0]()
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space

        self.agent_conns, self.env_conns = \
                zip(*[mp.Pipe() for _ in range(num_of_envs)])
        self.ps = []

        for agent_conn, env_conn, fn in zip(self.agent_conns, self.env_conns, env_fns):
            proc = mp.Process(target=worker,
                    args= (env_conn, agent_conn,CloudpickleWrapper(fn)))
            self.ps.append(proc)
        for p in self.ps:
            p.daemon = True
            p.start()

        for env_conn in self.env_conns:
            env_conn.close()

    def step_async(self, actions):
        if self.waiting:
            raise AlreadySteppingError
        self.waiting = True

        for agent_conn, action in zip(self.agent_conns, actions):
            agent_conn.send(('step', action))

    def step_wait(self):
        if not self.waiting:
            raise NotSteppingError
        self.waiting = False

        results = [agent_conn.recv() for agent_conn in self.agent_conns]
        obss, rets, dones, infos = zip(*results)
        return np.stack(obss), np.stack(rets), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for agent_conn in self.agent_conns:
            agent_conn.send(('reset', None))
        return np.stack([agent_conn.recv() for agent_conn in self.agent_conns]).squeeze()

    def sample(self):
        for agent_conn in self.agent_conns:
            agent_conn.send(('sample', None))

        results = [agent_conn.recv() for agent_conn in self.agent_conns]
        obss, rets, dones, infos = zip(*results)
        return np.stack(obss), np.stack(rets), np.stack(dones), infos

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for agent_conn in self.agent_conns:
                agent_conn.recv()
        for agent_conn in self.agent_conns:
            agent_conn.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def make_mp_envs(env_id, num_env, seed, start_idx=0):
    def make_env(rank):
        def fn():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)
            return env
        return fn
    return SubprocVecEnv([make_env(i + start_idx) for i in range(num_env)])

if __name__ == '__main__':
    # Test the performance of multi envs
    envs = MultiEnv('Ant-v3', 2)
    obs = envs.reset()
    import time
    s = time.perf_counter()
    for i in range(1000):
        obs, rets, dones, infos = envs.step(envs.sample())
    e = time.perf_counter()
    print(f"Time for one process: {e - s}")

    envs = make_mp_envs('Ant-v3', 2, 0)
    obs = envs.reset()
    s = time.perf_counter()
    for i in range(1000):
        obs, rets, dones, infos = envs.sample()
    e = time.perf_counter()
    print(f"Time for multi process: {e - s}")
