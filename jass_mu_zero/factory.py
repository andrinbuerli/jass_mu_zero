from pathlib import Path

import numpy as np

from jass_mu_zero.agent.agent import CppAgent
from jass_mu_zero.agent.agent_by_network_cpp import AgentByNetworkCpp
from jass_mu_zero.environment.networking.worker_config import WorkerConfig
from jass_mu_zero.observation.features_conv_cpp import FeaturesSetCppConv
from jass_mu_zero.observation.features_cpp_conv_cheating import FeaturesSetCppConvCheating
from jass_mu_zero.observation.features_set_cpp import FeaturesSetCpp


def get_agent(config: WorkerConfig, network, greedy=False, force_local=False) -> CppAgent:
    if config.agent.type == "mu-zero-mcts":
        from jass_mu_zero.agent.agent_mu_zero_mcts import AgentMuZeroMCTS
        return AgentMuZeroMCTS(
            network=network,
            feature_extractor=config.network.feature_extractor,
            iterations=config.agent.iterations,
            c_1=config.agent.c_1,
            c_2=config.agent.c_2,
            dirichlet_alpha=config.agent.dirichlet_alpha,
            dirichlet_eps=config.agent.dirichlet_eps if not greedy else 0.0,
            temperature=config.agent.temperature if not greedy else 5e-2,
            discount=config.agent.discount,
            mdp_value=config.agent.mdp_value,
            virtual_loss=config.agent.virtual_loss,
            n_search_threads=config.agent.n_search_threads,
            use_terminal_function=config.agent.terminal_func,
        )
    elif config.agent.type == "policy":
        from jass_mu_zero.agent.agent_policy import AgentPolicy
        return AgentPolicy(
            network=network,
            feature_extractor=config.network.feature_extractor,
            temperature=config.agent.temperature if not greedy else 5e-2,
        )
    elif config.agent.type == "dqn":
        from jass_mu_zero.agent.agent_dqn import AgentDQN
        return AgentDQN(model_path=str(Path(__file__).resolve().parent.parent / "resources" / "dqn.pt"))
    elif config.agent.type == "value":
        from jass_mu_zero.agent.agent_value import AgentValue
        return AgentValue(
            network=network,
            feature_extractor=config.network.feature_extractor,
            mdp_value=config.agent.mdp_value,
            temperature=config.agent.temperature if not greedy else 5e-2,
        )
    elif config.agent.type == "dmcts":
        if force_local:
            import jassmlcpp
            return jassmlcpp.agent.JassAgentDMCTSFullCpp(
                hand_distribution_policy=jassmlcpp.mcts.RandomHandDistributionPolicyCpp(),
                node_selection_policy=jassmlcpp.mcts.UCTPolicyFullCpp(exploration=np.sqrt(2)),
                reward_calculation_policy=jassmlcpp.mcts.RandomRolloutPolicyFullCpp(),
                nr_determinizations=config.agent.nr_determinizations,
                nr_iterations=config.agent.iterations,
                threads_to_use=config.agent.threads_to_use
            )
        else:
            if hasattr(config.agent, 'large') and config.agent.large:
                return AgentByNetworkCpp(url="http://localhost:9894/dmcts-large")
            else:
                return AgentByNetworkCpp(url="http://localhost:9898/dmcts")
    elif config.agent.type == "i-dmcts":
            return AgentByNetworkCpp(url="http://jass-agent.abiz.ch/theseus")
    elif config.agent.type == "i-dmcts-2":
            return AgentByNetworkCpp(url="http://localhost:5500/theseus")
    elif config.agent.type == "f-i-dmcts":
            return AgentByNetworkCpp(url="http://jass-agent.abiz.ch/tiresias")
    elif config.agent.type == "f-i-dmcts-2":
            return AgentByNetworkCpp(url="http://localhost:5500/tiresias")
    elif config.agent.type == "mcts":
        if force_local:
            import jassmlcpp
            return jassmlcpp.agent.JassAgentMCTSFullCpp(
                nr_iterations=config.agent.iterations,
                exploration=1.5
            )
        else:
            if hasattr(config.agent, 'large') and config.agent.large:
                return AgentByNetworkCpp(url="http://localhost:9893/mcts-large", cheating=True)
            else:
                return AgentByNetworkCpp(url="http://localhost:9899/mcts", cheating=True)
    elif config.agent.type == "random":
        if force_local:
            import jassmlcpp
            return jassmlcpp.agent.JassAgentRandomCpp()
        else:
            return AgentByNetworkCpp(url="http://localhost:9896/random")

    raise NotImplementedError(f"Agent type {config.agent.type} is not implemented.")


def get_network(config: WorkerConfig, network_path: str = None):
    if config.network.type == "resnet":
        from jass_mu_zero.mu_zero.network.resnet import MuZeroResidualNetwork
        network = MuZeroResidualNetwork(
            observation_shape=config.network.feature_extractor.FEATURE_SHAPE,
            action_space_size=config.network.action_space_size,
            num_blocks_representation=config.network.num_blocks_representation,
            fcn_blocks_representation=config.network.fcn_blocks_representation,
            num_blocks_dynamics=config.network.num_blocks_dynamics,
            fcn_blocks_dynamics=config.network.fcn_blocks_dynamics,
            num_blocks_prediction=config.network.num_blocks_prediction,
            num_channels=config.network.num_channels,
            reduced_channels_reward=config.network.reduced_channels_reward,
            reduced_channels_value=config.network.reduced_channels_value,
            reduced_channels_policy=config.network.reduced_channels_policy,
            fc_reward_layers=config.network.fc_reward_layers,
            fc_value_layers=config.network.fc_value_layers,
            fc_policy_layers=config.network.fc_policy_layers,
            fc_player_layers=config.network.fc_player_layers,
            fc_hand_layers=config.network.fc_hand_layers,
            fc_terminal_state_layers=config.network.fc_terminal_state_layers,
            support_size=config.network.support_size,
            players=config.network.players,
            mask_private=config.optimization.mask_private,
            mask_valid=config.optimization.mask_valid,
            fully_connected=config.network.fully_connected,
            network_path=network_path
        )

        return network

    raise NotImplementedError(f"Network type {config.network.type} is not implemented.")


def get_optimizer(config: WorkerConfig):
    import tensorflow_addons as tfa

    if config.optimization.optimizer == "adam":
        lr = config.optimization.learning_rate

        return tfa.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=config.optimization.weight_decay,
            beta_1=config.optimization.adam_beta1,
            beta_2=config.optimization.adam_beta2,
            epsilon=config.optimization.adam_epsilon)
    elif config.optimization.optimizer == "sgd":
        return tfa.optimizers.SGDW(
            learning_rate=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
            momentum=config.optimization.adam_beta1,
            nesterov=True)

    raise NotImplementedError(f"Optimizer {config.optimization.optimizer} is not implemented.")

def get_opponent(type: str) -> CppAgent:
    if type == "dmcts":
        return AgentByNetworkCpp(url="http://localhost:9898/dmcts")
    if type == "dmcts-50":
        return AgentByNetworkCpp(url="http://localhost:9895/dmcts-50")
    if type == "dmcts-large":
        return AgentByNetworkCpp(url="http://localhost:9894/dmcts-large")
    if type == "mcts":
        return AgentByNetworkCpp(url="http://localhost:9899/mcts", cheating=True)
    if type == "mcts-large":
        return AgentByNetworkCpp(url="http://localhost:9893/mcts-large", cheating=True)
    elif type == "random":
        return AgentByNetworkCpp(url="http://localhost:9896/random")
    raise NotImplementedError(f"Opponent type {type} is not implemented.")


def get_features(type: str) -> FeaturesSetCpp:
    if type == "cnn-full":
        return FeaturesSetCppConv()
    if type == "cnn-full-cheating":
        return FeaturesSetCppConvCheating()
    raise NotImplementedError(f"Features type {type} is not implemented.")


