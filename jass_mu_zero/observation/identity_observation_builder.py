import jasscpp
from jass_gym.observation.conv_observation_builder import ObservationBuilder


class IdentityObservationBuilder(ObservationBuilder):
    shape = (1,)

    def __call__(self, obs: jasscpp.GameObservationCpp):
        return {
            "obs": obs,
            "next_player": obs.player
        }
