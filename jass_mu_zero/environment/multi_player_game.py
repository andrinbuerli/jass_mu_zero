from collections import defaultdict
from typing import Callable

import gym
import numpy as np
from jass.game.const import team

from jass_mu_zero.agent.agent_full_action_space import AgentFullActionSpace


class MultiPlayerGame:
    """
    Wrapper around a rllib implementation of a finite sequential multiplayer game
    """

    def __init__(self, env: gym):
        self.env = env

    def play_rounds(self, get_agent: Callable[[int], AgentFullActionSpace], n=1):
        observations, rewards, actions = [], [], []
        action_probs, action_values = [], []

        for i in range(n):
            done = False
            obs = self.env.reset()
            try:
                while not done:
                    agentid = obs["observations"]["next_player"]
                    obs = obs["observations"]["obs"]

                    agent = get_agent(agentid)
                    action, probs, values = agent.action(obs)

                    actions.append(action)
                    action_probs.append(probs), action_values.append(values)
                    observations.append(obs)

                    obs, reward, done, info = self.env.step(action)

                    r = np.zeros(2)
                    r[team[agentid]] = reward

                    rewards.append(r)
            except Exception as e:
                print(e)

        return np.array(observations), np.array(rewards), np.array(actions), \
               np.array(action_probs), np.array(action_values)

    @staticmethod
    def _flatten_chronologically(data: dict, timesteps: list) -> list:
        flat = [y for x in data.values() for y in x]
        data = map(lambda x: x[1], sorted(zip(timesteps, flat), key=lambda x: x[0]))
        return list(data)
