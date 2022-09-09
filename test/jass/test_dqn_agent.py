import jasscpp

from jass_mu_zero.factory import get_agent
from test.util import get_test_config


def test_inference():
    config = get_test_config()
    config.agent.type = "dqn"
    testee = get_agent(config, None)

    obs = jasscpp.observation_from_state(jasscpp.GameSimCpp().state, -1)

    testee.action_trump(obs)