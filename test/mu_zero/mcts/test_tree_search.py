from pathlib import Path

import jasscpp

from jass_mu_zero.factory import get_network
from jass_mu_zero.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy
from jass_mu_zero.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from jass_mu_zero.mu_zero.mcts.min_max_stats import MinMaxStats
from jass_mu_zero.mu_zero.mcts.tree_search import ALPV_MCTS
from jass_mu_zero.mu_zero.network.buffering_network import BufferingNetwork
from jass_mu_zero.observation.features_conv_cpp import FeaturesSetCppConv
from test.util import get_test_config


def test_single_simulation():
    config = get_test_config()
    network = get_network(config)

    stats = MinMaxStats()

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        value_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1
    )

    testee.run_simulation()

    assert testee.root.visits == 1


def test_multiple_simulations():
    config = get_test_config()
    network = get_network(config)

    stats = MinMaxStats()

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        value_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1
    )

    testee.run_simulations_sync(100)

    assert testee.root.visits == 100


def test_multiple_simulations_async_single_thread():
    config = get_test_config()
    network = get_network(config)

    stats = MinMaxStats()

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        value_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=10,
        n_search_threads=1
    )

    testee.run_simulations_async(10)

    assert testee.root.visits == 10

def test_multiple_simulations_async_multi_thread():
    config = get_test_config()
    network = get_network(config)

    stats = MinMaxStats()

    n_search_threads = 4
    buffered_network = BufferingNetwork(network, buffer_size=n_search_threads, timeout=0.1)

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=buffered_network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        value_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=10,
        n_search_threads=n_search_threads
    )

    testee.run_simulations_async(20)

    assert testee.root.visits == 20

    del buffered_network


def test_multiple_simulations_async_multi_thread_concurrency_check():
    config = get_test_config()
    network = get_network(config)

    stats = MinMaxStats()

    n_search_threads = 4
    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        value_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=0, # provoke concurrency issues
        n_search_threads=n_search_threads
    )

    testee.run_simulations_async(1000)

    assert testee.root.visits == 1000

    del testee


def test_get_rewards():
    config = get_test_config()
    network = get_network(config)

    stats = MinMaxStats()

    n_search_threads = 4
    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        value_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=0, # provoke concurrency issues
        n_search_threads=n_search_threads
    )

    testee.run_simulations_async(1000)

    prob, q_value = testee.get_result()

    assert prob.shape == (43,)
    assert q_value.shape == (43, 2)


def test_get_rewards_lots_threads():
    config = get_test_config()
    network = get_network(config)

    stats = MinMaxStats()

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    n_search_threads = 16

    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        value_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=10, # provoke concurrency issues
        n_search_threads=n_search_threads
    )
    testee.run_simulations_async(20)

    prob, q_value = testee.get_result()

    assert prob.shape == (43,)
    assert q_value.shape == (43, 2)


def test_sync_consistency():
    config = get_test_config()
    network = get_network(config)

    network.load(Path(__file__).parent.parent.parent / "resources" / "imperfect_resnet_random.pd", from_graph=True)

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.0,
            dirichlet_alpha=0.3,
            discount=1)

    for _ in range(5):
        obs = jasscpp.GameObservationCpp()
        obs.player = 1
        stats = MinMaxStats()
        testee1 = ALPV_MCTS(
            observation=obs,
            node_selection=tree_policy,
            value_calc=LatentValueCalculationPolicy(),
            mdp_value=False,
            stats=stats,
            discount=1,
            virtual_loss=10,
            n_search_threads=1,
            store_trajectory_actions=True
        )

        testee1.run_simulations_sync(50)

        result1 = testee1.get_result()

        obs = jasscpp.GameObservationCpp()
        obs.player = 1
        stats = MinMaxStats()
        testee2 = ALPV_MCTS(
            observation=obs,
            node_selection=tree_policy,
            value_calc=LatentValueCalculationPolicy(),
            mdp_value=False,
            stats=stats,
            discount=1,
            store_trajectory_actions=True
        )

        testee2.run_simulations_sync(50)

        result2 = testee2.get_result()

        print((result1[1] - result2[1]).max())
        assert all([x == y for x, y in zip(testee1.trajectory_actions, testee2.trajectory_actions)])

    del testee1, testee2


def test_sync_vs_async_consistency():
    config = get_test_config()
    network = get_network(config)

    network.load(Path(__file__).parent.parent.parent / "resources" / "imperfect_resnet_random.pd", from_graph=True)

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.0,
            dirichlet_alpha=0.3,
            discount=1)

    for _ in range(5):
        obs = jasscpp.GameObservationCpp()
        obs.player = 1
        stats = MinMaxStats()
        testee1 = ALPV_MCTS(
            observation=obs,
            node_selection=tree_policy,
            value_calc=LatentValueCalculationPolicy(),
            mdp_value=False,
            stats=stats,
            discount=1,
            virtual_loss=10,
            n_search_threads=1
        )

        testee1.run_simulations_async(50)

        result1 = testee1.get_result()

        obs = jasscpp.GameObservationCpp()
        obs.player = 1
        stats = MinMaxStats()
        testee2 = ALPV_MCTS(
            observation=obs,
            node_selection=tree_policy,
            value_calc=LatentValueCalculationPolicy(),
            mdp_value=False,
            stats=stats,
            discount=1
        )

        testee2.run_simulations_sync(50)

        result2 = testee2.get_result()

        print((result1[1] - result2[1]).max())
        assert all([x == y for x, y in zip(testee1.trajectory_actions, testee2.trajectory_actions)])

    del testee1, testee2


def test_sync_vs_async_consistency_multi_threads():
    config = get_test_config()
    network = get_network(config)

    network.load(Path(__file__).parent.parent.parent / "resources" / "imperfect_resnet_random.pd", from_graph=True)

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.0,
            dirichlet_alpha=0.3,
            discount=1)

    for _ in range(5):
        obs = jasscpp.GameObservationCpp()
        obs.player = 1
        stats = MinMaxStats()
        testee1 = ALPV_MCTS(
            observation=obs,
            node_selection=tree_policy,
            value_calc=LatentValueCalculationPolicy(),
            mdp_value=False,
            stats=stats,
            discount=1,
            virtual_loss=10,
            n_search_threads=4,
            store_trajectory_actions=True
        )

        testee1.run_simulations_async(100)

        result1 = testee1.get_result()

        obs = jasscpp.GameObservationCpp()
        obs.player = 1
        stats = MinMaxStats()
        testee2 = ALPV_MCTS(
            observation=obs,
            node_selection=tree_policy,
            value_calc=LatentValueCalculationPolicy(),
            mdp_value=False,
            stats=stats,
            discount=1,
            store_trajectory_actions=True
        )

        testee2.run_simulations_sync(100)

        result2 = testee2.get_result()

        assert (result1[1] - result2[1]).max() < 5

    del testee1, testee2
