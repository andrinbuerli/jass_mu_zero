from jass_mu_zero.environment.networking.worker_config import WorkerConfig
from jass_mu_zero.observation.features_conv_cpp import FeaturesSetCppConv
from jass_mu_zero.observation.features_cpp_conv_cheating import FeaturesSetCppConvCheating


def get_test_config(cheating=False):
    config = WorkerConfig(features=FeaturesSetCppConv() if not cheating else FeaturesSetCppConvCheating())
    config.network.type = "resnet"
    config.network.num_blocks_representation = 1
    config.network.fcn_blocks_representation = 0
    config.network.fcn_blocks_representation = 1
    config.network.num_blocks_prediction = 0
    config.network.num_blocks_prediction = 0
    config.network.num_channels = 128
    config.network.support_size = 100
    return config


def get_test_resnet():
    from jass_mu_zero.mu_zero.network.resnet import MuZeroResidualNetwork
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=43,
        num_blocks_representation=1,
        fcn_blocks_representation=0,
        num_blocks_dynamics=1,
        fcn_blocks_dynamics=0,
        num_blocks_prediction=0,
        num_channels=256,
        reduced_channels_reward=128,
        reduced_channels_value=1,
        reduced_channels_policy=128,
        fc_reward_layers=[256],
        fc_value_layers=[256],
        fc_policy_layers=[256],
        fc_hand_layers=[256],
        fc_player_layers=[256],
        fc_terminal_state_layers=[256],
        mask_valid=False,
        mask_private=False,
        support_size=100,
        players=4,
        fully_connected=False
    )
    return network
