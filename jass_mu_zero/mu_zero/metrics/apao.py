import gc
from copy import deepcopy

import numpy as np
from jass.game.const import team

from jass_mu_zero.agent.agent import CppAgent
from jass_mu_zero.environment.multi_player_game import MultiPlayerGame
from jass_mu_zero.environment.networking.worker_config import WorkerConfig
from jass_mu_zero.factory import get_agent, get_opponent
from jass_mu_zero.mu_zero.metrics.base_async_metric import BaseAsyncMetric
from jass_mu_zero.mu_zero.network.network_base import AbstractNetwork
from jass_mu_zero.observation.identity_observation_builder import IdentityObservationBuilder


def _play_single_game_(i, agent: CppAgent, opponent: CppAgent):
    from jass_gym.env.jass_single_agent_env import SchieberJassSingleAgentEnv
    first_team = np.random.choice([True, False])

    game = MultiPlayerGame(env=SchieberJassSingleAgentEnv(observation_builder=IdentityObservationBuilder()))
    if first_team:
        _, rewards, _, _, _ = game.play_rounds(get_agent=lambda key: {0: agent, 1: opponent}[team[key]], n=4)

        points = np.array([np.sum(rewards[:, 0]), np.sum(rewards[:, 1])])
        points = np.mean(points[0] / points.sum())
    else:
        _, rewards, _, _, _ = game.play_rounds(get_agent=lambda key: {1: agent, 0: opponent}[team[key]], n=4)

        points = np.array([np.sum(rewards[:, 0]), np.sum(rewards[:, 1])])
        points = np.mean(points[1] / points.sum())

    del game, agent, opponent
    gc.collect()

    return points


class APAO(BaseAsyncMetric):

    def get_params(self, thread_nr: int, network: AbstractNetwork, init_vars=None) -> []:
        return thread_nr, get_agent(self.worker_config, network, greedy=True), get_opponent(self.opponent_name)

    def __init__(self, opponent_name: str, worker_config: WorkerConfig, network_path: str, parallel_threads: int,
                 only_policy=False):

        if only_policy:
            worker_config = deepcopy(worker_config)
            worker_config.agent.type = "policy"

        self.only_policy = only_policy

        self.opponent_name = opponent_name
        super().__init__(worker_config, network_path, parallel_threads, _play_single_game_)

    def get_name(self):
        if self.only_policy:
            return f"apa_{self.opponent_name}_raw_policy"
        else:
            return f"apa_{self.opponent_name}"
