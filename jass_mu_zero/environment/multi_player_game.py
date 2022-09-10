from collections import defaultdict
from typing import Callable

from ray.rllib import MultiAgentEnv

from jass_mu_zero.agent.agent_full_action_space import AgentFullActionSpace


class MultiPlayerGame:
    """
    Wrapper around a rllib implementation of a finite sequential multiplayer game
    """

    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def play_rounds(self, get_agent: Callable[[int], AgentFullActionSpace], n=1):
        observations, rewards, actions = defaultdict(list), defaultdict(list), defaultdict(list)
        action_probs, action_values = defaultdict(list), defaultdict(list)

        for i in range(n):
            done = {"__al__": False}
            obs = self.env.reset()
            while done["__al__"] is False:
                agentid = list(obs.keys())[0]
                agent = get_agent(agentid)
                obs = list(obs["observations"].items())[0]

                action, probs, values = agent.action(obs)

                actions[agentid].append(action)
                action_probs[agentid].append(probs), action_values[agentid].append(values)
                observations[agentid].append(obs)

                obs, reward, done, _ = self.env.step(action)

                rewards[agentid].append(reward)

        return observations, rewards, actions, action_probs, action_values
