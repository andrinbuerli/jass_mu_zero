# HSLU
#
# Created by Thomas Koller on 27.10.2020
#

import numpy as np
import torch
from jass.game.const import PUSH, PUSH_ALT
from jasscpp import GameObservationCpp, RuleSchieberCpp

from jass_mu_zero.agent.agent import CppAgent
from jass_mu_zero.observation.features_conv_cpp import FeaturesSetCppConv


class AgentDQN(CppAgent):
    """
    DQN trained agent
    """

    def __init__(self, model_path: str):
        self.model = torch.jit.load(model_path)
        self.rule = RuleSchieberCpp()
        self.features = FeaturesSetCppConv()

    def action_play_card(self, obs: GameObservationCpp) -> int:
        return self.get_action(obs)

    def action_trump(self, obs: GameObservationCpp) -> int:
        action = self.get_action(obs) - 36
        if action == PUSH_ALT:
            action = PUSH

        return action

    def get_action(self, obs):
        valid_actions = self.rule.get_valid_cards_from_obs(obs)
        obs = self.features.convert_to_features(obs, self.rule)
        masked_policy = self.model.inference(torch.tensor(obs.reshape(self.features.FEATURE_SHAPE)[None]),
                                             torch.tensor(valid_actions[None])).detach().numpy()[0]

        return int(np.argmax(masked_policy))
