import logging

import requests
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import PUSH, PUSH_ALT, TRUMP_FULL_OFFSET, card_ids
from jass.game.game_observation import GameObservation
from jass.service.player_service_route import PLAY_CARD_PATH_PREFIX, SELECT_TRUMP_PATH_PREFIX, SEND_INFO_PREFIX


class AgentByNetwork(Agent):
    """
    Forwards the request to a player service. Used for locally playing against deployed services.

    A random agent is used as standing player, if the service does not answer within a timeout.
    """

    def __init__(self, url, timeout=120):
        self._logger = logging.getLogger(__name__)
        self._standin_player = AgentRandomSchieber()
        self._base_url = url
        self._url_info = self._base_url + SEND_INFO_PREFIX
        self._url_trump = self._base_url + SELECT_TRUMP_PATH_PREFIX
        self._url_play = self._base_url + PLAY_CARD_PATH_PREFIX
        self._timeout = timeout

    def action_trump(self, obs: GameObservation) -> int:
        data = obs.to_json()
        data['gameId'] = 0
        # noinspection PyBroadException
        try:
            response = requests.post(self._url_trump, json=data, timeout=self._timeout)
            response_data = response.json()
            trump = int(response_data['trump'])
            return trump
        except Exception as e:
            self._logger.error(f'No response from network player, using standin player ({e})')
            return self._standin_player.action_trump(obs)

    # noinspection PyBroadException
    def action_play_card(self, obs: GameObservation) -> int:
        data = obs.to_json()
        data['gameId'] = 0
        try:
            response = requests.post(self._url_play, json=data, timeout=self._timeout)
            response_data = response.json()
            card = response_data['card']
            card_id = card_ids[card]
            return card_id
        except Exception as e:
            self._logger.error(f'No response from network player, using standin player ({e})')
            return self._standin_player.action_play_card(obs)

    def action(self, obs: GameObservation) -> int:
        if obs.trump == -1:
            trump = self.action_trump(obs)
            if trump == PUSH:
                trump = PUSH_ALT
            action = TRUMP_FULL_OFFSET + trump
        else:
            action = self.action_play_card(obs)

        return action, None, None
