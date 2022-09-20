from typing import Union

import jasscpp
import numpy as np
from jass.game.const import next_player, TRUMP_FULL_OFFSET, TRUMP_FULL_P, card_values
from jass.game.rule_schieber import RuleSchieber

from jass_mu_zero.mu_zero.mcts.min_max_stats import MinMaxStats
from jass_mu_zero.mu_zero.mcts.node import Node
from jass_mu_zero.mu_zero.network.network_base import AbstractNetwork
from jass_mu_zero.mu_zero.network.support_conversion import support_to_scalar
from jass_mu_zero.observation.features_set_cpp import FeaturesSetCpp


class LatentNodeSelectionPolicy:
    """
    Node selection policy in a learned latent space.
    Implemented according to paper: https://arxiv.org/abs/1911.08265
    """

    def __init__(
            self,
            c_1: float,
            c_2: float,
            feature_extractor: FeaturesSetCpp,
            network: AbstractNetwork,
            discount: float,
            dirichlet_eps: float = 0.25,
            dirichlet_alpha: float = 0.3,
            mdp_value: bool = False,
            use_terminal_function: bool = False):
        """
        Initialise the selection policy
        :param c_1: c_1 parameter of PUCT heuristic
        :param c_2: c_2 parameter of PUCT heuristic
        :param feature_extractor: feature extractor compatible with provided network
        :param network: trained network which can project a game state to the latent space and change adapt it according
                        to selected actions
        :param discount: discount to use when bootstrapping the value function
        :param dirichlet_eps: dirichlet epsilon
        :param dirichlet_alpha: dirichlet alpha
        :param mdp_value: boolean indicating if mdp value should be used
        :param use_terminal_function: boolean indicating if the estimated terminal function should be used
        """

        self.use_terminal_function = use_terminal_function
        self.mdp_value = mdp_value
        self.discount = discount
        self.c_2 = c_2
        self.c_1 = c_1
        self.network = network
        self.dirichlet_alpha = dirichlet_alpha * np.ones(43)
        self.dirichlet_eps = dirichlet_eps
        self.feature_extractor = feature_extractor
        self.rule = RuleSchieber()

    def tree_policy(
            self,
            observation: jasscpp.GameObservationCpp,
            root_node: Node,
            stats: MinMaxStats,
            virtual_loss=0,
            observation_feature_format=None) -> Node:
        """
        Select next node for expansion in the latent space
        :param observation: observation at the root node
        :param root_node: root node
        :param stats: statistics of the in memory search tree
        :param virtual_loss: virtual loss to use
        :param observation_feature_format: format of the observation features,
                if this parameter not none, then observation is a feature map from a supervised dataset
        :return: selected node
        """

        node = root_node
        while True:
            with node.lock: # ensures that node and children not currently locked, i.e. being expanded
                node.visits += virtual_loss

            valid_actions = node.valid_actions

            assert valid_actions.sum() > 0, 'Error in valid actions'

            children = node.children_for_action(valid_actions)

            assert len(children) > 0, f'Error no children for valid actions {valid_actions}, {vars(node)}'

            exploration = np.array([self._exploration_term(x) for x in children])
            exploitation = np.array([self._exploitation_term(x, stats) for x in children])

            puct = exploitation + exploration
            i_max = np.argmax(puct)
            child = children[i_max]

            for c in children:
                c.avail += 1

            is_terminal_state = self._get_is_terminal_state(child)

            with node.lock:
                with child.lock:
                    not_expanded = child.prior is None
                    if not_expanded or is_terminal_state:
                        if not_expanded:
                            child.visits += virtual_loss
                            child.value, child.reward, child.prior, child.predicted_player, _, child.is_post_terminal, child.hidden_state = \
                                self.network.recurrent_inference(node.hidden_state, np.array([[child.action]]), all_preds=True)
                            self._expand_node(child)
                        break

            node = child

        return child

    def init_node(self, node: Node, observation: Union[jasscpp.GameStateCpp, jasscpp.GameObservationCpp], observation_feature_format=None):
        if node.is_root():
            rule = jasscpp.RuleSchieberCpp()
            if observation_feature_format is not None:
                node.valid_actions = observation_feature_format.valid_actions
            elif type(observation) == jasscpp.GameStateCpp:
                node.valid_actions = rule.get_full_valid_actions_from_state(observation)
            else:
                node.valid_actions = rule.get_full_valid_actions_from_obs(observation)

            assert (node.valid_actions >= 0).all(), 'Error in valid actions'

            if observation_feature_format is not None:
                features = observation[None]
            else:
                features = self.feature_extractor.convert_to_features(observation, rule)[None]
            node.value, node.reward, node.prior, node.predicted_player, _, node.is_post_terminal, node.hidden_state =\
                self.network.initial_inference(features, all_preds=True)
            self._expand_node(node)

            valid_idxs = np.where(node.valid_actions)[0]
            eta = np.random.dirichlet(self.dirichlet_alpha[:len(valid_idxs)])
            node.prior[valid_idxs] = (1 - self.dirichlet_eps) * node.prior[valid_idxs] + self.dirichlet_eps * eta

    def _exploration_term(self, child: Node):
        P_s_a = child.parent.prior[child.action]
        prior_weight = (np.sqrt(child.avail) / (1 + child.visits)) * (
                    self.c_1 + np.log((child.avail + self.c_2 + 1) / self.c_2))
        exploration_term = P_s_a * prior_weight

        return exploration_term

    def _exploitation_term(self, child: Node, stats: MinMaxStats):
        if child.visits > 0:
            with child.lock:
                next_player = child.parent.next_player
                q = (child.value_sum[next_player] / child.visits)
                assert len(child.reward.shape) == 1, f'shape: {child.reward.shape}'
                q_value = (child.reward[next_player] + self.discount * q) \
                    if self.mdp_value else q

                q_value = stats.normalize(q_value)
        else:
            q_value = 0

        return q_value

    def _get_is_terminal_state(self, child):
        is_terminal_state = child.is_post_terminal > 0.5 if (
                    self.use_terminal_function and child.is_post_terminal is not None) else False
        return is_terminal_state

    def _expand_node(self, node: Node):
        node.value, node.reward, node.prior, node.predicted_player, node.is_post_terminal = \
            [x.numpy().squeeze() for x in [node.value, node.reward, node.prior, node.predicted_player, node.is_post_terminal]]

        value_support_size = node.value.shape[-1]
        node.value = support_to_scalar(distribution=node.value, min_value=-value_support_size//2).numpy()

        if self._get_is_terminal_state(node) and self.mdp_value:
            node.value = np.zeros_like(node.value)

        reward_support_size = node.reward.shape[-1]
        node.reward = support_to_scalar(distribution=node.reward, min_value=-reward_support_size//2).numpy()

        # add edges for all children
        for action in node.missing_actions(node.valid_actions):
            node.add_child(
                action=action,
                next_player=node.predicted_player.argmax(),
                trump=-1,
                mask_invalid=False)
