import gym_super_mario_bros as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
env = gym.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env,SIMPLE_MOVEMENT)
obs = env.reset()

while True:
    _, _, done, _ = env.step(env.action_space.sample())
    import ipdb; ipdb.set_trace()
    env.render()
    if done:
        break
