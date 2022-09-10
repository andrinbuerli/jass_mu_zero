import abc
import numpy as np
import logging

from jass.game.const import PUSH, PUSH_ALT

from jasscpp import GameObservationCpp


class AgentFullActionSpace(CppAgent):

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def action_trump(self, obs: GameObservationCpp) -> int:
        distribution, values = self.get_play_action_probs_and_values(obs)
        trump_distribution = self._heat_prob(distribution[36:])

        if obs.forehand == -1:
            # if forehand is not yet set, we are the forehand player and can select trump or push
            action_trump = np.random.choice(np.arange(0, 7, 1), p=trump_distribution)
            if action_trump == PUSH_ALT:
                action_trump = PUSH
        else:
            # push is not allowed
            trump_distribution = self._heat_prob(distribution[36:-1])
            action_trump = np.random.choice(np.arange(0, 6, 1), p=trump_distribution[0:6])

        logging.debug(f'Trump response: {action_trump}')

        return int(action_trump)

    def action_play_card(self, obs: GameObservationCpp) -> int:
        distribution, values = self.get_play_action_probs_and_values(obs)
        card_distribution = self._heat_prob(distribution[:36])
        action_card = np.random.choice(np.arange(0, 36, 1), p=card_distribution)

        logging.debug(f'Card response: {action_card}')

        return int(action_card)

    def action(self, obs: GameObservationCpp) -> (int, np.array, np.array):
        distribution, values = self.get_play_action_probs_and_values(obs)
        card_distribution = self._heat_prob(distribution[:36])
        action = np.random.choice(np.arange(0, 43, 1), p=card_distribution)

        logging.debug(f'Action response: {action}')

        return int(action), distribution, values

    @abc.abstractmethod
    def get_play_action_probs_and_values(self, obs: GameObservationCpp) -> (np.array, np.array):
        """
        Determine the probability distribution and values over the next possible actions (card or trump).

        :param obs: the game state
        :return:
            (
                the probability distribution over the next actions (shape = [43] = [36 + 7]),
                a value estimate for each next action (shape = [43] = [36 + 7])
            )
        """
        raise NotImplementedError()

    def _heat_prob(self, prob, delta=1e-3):
        if prob.max() == 0:
            prob += delta

        prob_hot = prob ** (1 / self.temperature)

        if np.isnan(prob_hot).any():
            logging.warning("!! NAN encountered in prob norming !!")
            prob_hot = np.clip(prob_hot, 1e-10, 1)

        return prob_hot / prob_hot.sum()
