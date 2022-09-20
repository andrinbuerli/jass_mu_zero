from collections import defaultdict
from typing import Callable

import gym
import jasscpp
import numpy as np
from jass.game.const import team

from jass_mu_zero.agent.agent_full_action_space import AgentFullActionSpace


class MultiPlayerGame:
    """
    Wrapper around a gym implementation of a finite sequential multiplayer game.
    """

    def __init__(self, env: gym):
        self.env = env

    def play_rounds(self, get_agent: Callable[[int], AgentFullActionSpace], n=1):
        observations, rewards, actions = [], [], []
        action_probs, action_values = [], []

        for i in range(n):
            done = False
            obs = self.env.reset()
            while not done:
                agentid = obs["observations"]["next_player"]
                obs = obs["observations"]["obs"]

                agent = get_agent(agentid)
                assert obs.hand.sum() > 0, f"no hand... {self.env._game.state.hands}, {self.env._game.state}"
                action, probs, values = agent.action(obs)

                valid_actions = jasscpp.RuleSchieberCpp().get_full_valid_actions_from_obs(obs)
                assert action in np.flatnonzero(valid_actions), f"{agent}: {action}, {valid_actions}"

                actions.append(action)
                action_probs.append(probs), action_values.append(values)
                observations.append(obs)

                obs, _, done, info = self.env.step(action)

                reward = info["team_reward"]
                rewards.append(reward)

        return np.array(observations), np.array(rewards), np.array(actions), \
               np.array(action_probs), np.array(action_values)

    @staticmethod
    def _flatten_chronologically(data: dict, timesteps: list) -> list:
        flat = [y for x in data.values() for y in x]
        data = map(lambda x: x[1], sorted(zip(timesteps, flat), key=lambda x: x[0]))
        return list(data)
