import gym
from gym import spaces
import numpy as np


class TagPositiveWrapper(gym.Env):
    def __init__(self, env, adversary_policy=None, adversary_graph=None):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space[-1]
        self.action_space = env.action_space[-1]
        self.adversary_policy = adversary_policy
        self.obs_n = env.reset()
        self.adversary_graph = adversary_graph

    def step(self, action):
        action_n = [agent.action(obs) for agent, obs in zip(self.adversary_policy[:-1], self.obs_n[:-1])]
        action_n.append(action)
        next_state, reward, done, info = self.env.step(action_n)

        self.obs_n = next_state
        reward = reward[-1]
        done = all(done)
        info = info[-1]

        return next_state[-1], reward, done, info

    def reset(self):
        self.obs_n = self.env.reset()

        return self.obs_n[-1]


