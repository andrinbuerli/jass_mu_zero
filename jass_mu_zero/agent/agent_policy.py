import jasscpp
import numpy as np
from jasscpp import GameObservationCpp

from jass_mu_zero.agent.agent_full_action_space import AgentFullActionSpace
from jass_mu_zero.observation.features_cpp_conv_cheating import FeaturesSetCppConvCheating
from jass_mu_zero.observation.features_set_cpp import FeaturesSetCpp
from jass_mu_zero.mu_zero.network.network_base import AbstractNetwork


class AgentPolicy(AgentFullActionSpace):
    """
    Local policy agent to play the Schieber jass
    """

    def __init__(self,
                 network: AbstractNetwork,
                 feature_extractor: FeaturesSetCpp,
                 temperature=1.0):
        super().__init__(temperature=temperature)
        self.network = network
        self.feature_extractor = feature_extractor
        self.cheating_mode = type(feature_extractor) == FeaturesSetCppConvCheating
        self.rule = jasscpp.RuleSchieberCpp()

    def get_play_action_probs_and_values(self, obs: GameObservationCpp, feature_format=None) -> np.array:
        features = self.feature_extractor.convert_to_features(obs, self.rule)
        value, reward, policy, next_encoded_state = self.network.initial_inference(features[None])

        if type(obs) == GameObservationCpp:
            policy = policy.numpy().reshape(-1) * self.rule.get_full_valid_actions_from_obs(obs)
        else:
            policy = policy.numpy().reshape(-1) * self.rule.get_full_valid_actions_from_state(obs)

        values = np.ones((policy.shape[0], 2))  # using only the local policy does not calculate a value
        return policy, values
