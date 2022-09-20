import argparse
import logging
import sys
from pathlib import Path
from pprint import pprint

import tensorflow as tf
from jass.features.labels_action_full import LabelSetActionFull


sys.path.append("../")

from jass_mu_zero.log.wandb_logger import WandbLogger
from jass_mu_zero.mu_zero.metrics.save import SAVE
from jass_mu_zero.mu_zero.metrics.spkl import SPKL
from jass_mu_zero.mu_zero.metrics.vpkl import VPKL
from jass_mu_zero.mu_zero.metrics.sare import SARE
from jass_mu_zero.mu_zero.metrics.lse import LSE
from jass_mu_zero.environment.networking.worker_config import WorkerConfig
from jass_mu_zero.environment.networking.worker_connector import WorkerConnector
from jass_mu_zero.factory import get_network, get_features, get_optimizer
from jass_mu_zero.log.console_logger import ConsoleLogger
from jass_mu_zero.mu_zero.metrics.apao import APAO
from jass_mu_zero.mu_zero.metrics.visualise_game import GameVisualisation
from jass_mu_zero.mu_zero.metrics.metrics_manager import MetricsManager
from jass_mu_zero.mu_zero.replay_buffer.file_based_replay_buffer_from_folder import FileBasedReplayBufferFromFolder
from jass_mu_zero.mu_zero.replay_buffer.supervised_replay_buffer_from_folder import SupervisedReplayBufferFromFolder
from jass_mu_zero.mu_zero.trainer import MuZeroTrainer


class MuZeroTrainingCLI:
    @staticmethod
    def setup_args(parser: argparse.ArgumentParser):
        parser.add_argument(f'--file', default="settings.json")
        parser.add_argument(f'--log', default=False, action="store_true")
        parser.add_argument(f'--eager', default=False, action="store_true")

    @staticmethod
    def run(args):
        if args.eager:
            tf.config.experimental_run_functions_eagerly(True)

        worker_config = WorkerConfig()
        file_path = Path(__file__).parent.parent.parent / "resources" / args.file
        logging.info(f"Loading experiment from file {file_path}..")
        worker_config.load_from_json(str(file_path))

        pprint(worker_config.to_json())

        data_path = Path(worker_config.optimization.data_folder).resolve() / f"{worker_config.timestamp}"
        logging.info(f"creating result directory at {data_path}")
        data_path.mkdir(parents=True, exist_ok=True)
        worker_config.save_to_json(data_path / "worker_config.json")

        worker_config.network.feature_extractor = get_features(worker_config.network.feature_extractor)

        network = get_network(worker_config)
        network_path = data_path / "latest_network.pd"
        if network_path.exists():
            try:
                network.load(network_path)
            except Exception as e:
                logging.warning(f"could not restore network: {e}")
                network.save(network_path)
        else:
            network.save(network_path)

        network.summary()

        if worker_config.optimization.supervised_targets:
            replay_buffer = SupervisedReplayBufferFromFolder(
                features=worker_config.network.feature_extractor,
                batch_size=worker_config.optimization.batch_size,
                nr_of_batches=worker_config.optimization.updates_per_step,
                max_trajectory_length=worker_config.optimization.trajectory_length,
                min_trajectory_length=worker_config.optimization.min_trajectory_length,
                cache_path=data_path,
                mdp_value=worker_config.agent.mdp_value,
                valid_policy_target=worker_config.optimization.valid_policy_target,
                gamma=worker_config.agent.discount,
                start_sampling=False,
                td_error=worker_config.optimization.value_td_5_step)
        else:
            replay_buffer = FileBasedReplayBufferFromFolder(
                max_buffer_size=worker_config.optimization.max_buffer_size,
                batch_size=worker_config.optimization.batch_size,
                nr_of_batches=worker_config.optimization.updates_per_step,
                trajectory_length=worker_config.optimization.trajectory_length,
                game_data_folder=data_path / "game_data",
                clean_up_files=True,
                cache_path=data_path,
                mdp_value=worker_config.agent.mdp_value,
                gamma=worker_config.agent.discount,
                start_sampling=False,
                trajectory_data_folder=data_path / "trajectory_data",
                max_samples_per_episode=worker_config.optimization.max_samples_per_episode,
                min_non_zero_prob_samples=worker_config.optimization.min_non_zero_prob_samples,
                use_per=worker_config.optimization.use_per,
                value_based_per=worker_config.optimization.value_based_per,
                td_error=worker_config.optimization.value_td_5_step,)

        replay_buffer.restore(tree_from_file=worker_config.optimization.restore_buffer_tree_from_file)

        metrics = [
            APAO("dmcts", worker_config, str(network_path), parallel_threads=worker_config.optimization.apa_n_games),
            APAO("dmcts", worker_config, str(network_path), parallel_threads=worker_config.optimization.apa_n_games,
                 only_policy=True),
            APAO("random", worker_config, str(network_path), parallel_threads=worker_config.optimization.apa_n_games),
            SARE(
                samples_per_calculation=worker_config.optimization.batch_size,
                label_length=LabelSetActionFull.LABEL_LENGTH,
                worker_config=worker_config,
                network_path=str(network_path),
                n_steps_ahead=worker_config.optimization.log_n_steps_ahead,
                mdp_value=worker_config.agent.mdp_value
            ),
            SAVE(
                samples_per_calculation=worker_config.optimization.batch_size,
                label_length=LabelSetActionFull.LABEL_LENGTH,
                worker_config=worker_config,
                network_path=str(network_path),
                n_steps_ahead=worker_config.optimization.log_n_steps_ahead,
                mdp_value=worker_config.agent.mdp_value
            ),
            SPKL(
                samples_per_calculation=worker_config.optimization.batch_size,
                label_length=LabelSetActionFull.LABEL_LENGTH,
                worker_config=worker_config,
                network_path=str(network_path),
                n_steps_ahead=worker_config.optimization.log_n_steps_ahead
            ),
            # VPKL(
            #     samples_per_calculation=worker_config.optimization.batch_size,
            #     label_length=LabelSetActionFull.LABEL_LENGTH,
            #     worker_config=worker_config,
            #     network_path=str(network_path),
            #     n_steps_ahead=worker_config.optimization.log_n_steps_ahead
            # ),
            LSE(
                samples_per_calculation=worker_config.optimization.batch_size,
                label_length=LabelSetActionFull.LABEL_LENGTH,
                worker_config=worker_config,
                network_path=str(network_path),
                n_steps_ahead=worker_config.optimization.log_n_steps_ahead
            )
        ]

        if worker_config.optimization.log_visualisations:
            metrics += [
                GameVisualisation(
                    label_length=LabelSetActionFull.LABEL_LENGTH,
                    worker_config=worker_config,
                    network_path=str(network_path),
                    mdp_value=worker_config.agent.mdp_value
                )
            ]

        manager = MetricsManager(*metrics)

        if args.log:
            with open("/app/.wandbkey", "r") as f:
                api_key = f.read().rstrip()
            logger = WandbLogger(
                wandb_project_name=worker_config.log.projectname,
                group_name=worker_config.log.group,
                api_key=api_key,
                entity=worker_config.log.entity,
                run_name=f"{worker_config.log.group}-{worker_config.agent.type}-{worker_config.timestamp}",
                config=worker_config.to_json()
            )
        else:
            logger = ConsoleLogger({})

        optimizer = get_optimizer(worker_config)

        trainer = MuZeroTrainer(
            network=network,
            replay_buffer=replay_buffer,
            metrics_manager=manager,
            logger=logger,
            config=worker_config,
            value_loss_weight=worker_config.optimization.value_loss_weight,
            reward_loss_weight=worker_config.optimization.reward_loss_weight,
            policy_loss_weight=worker_config.optimization.policy_loss_weight,
            player_loss_weight=worker_config.optimization.player_loss_weight,
            hand_loss_weight=worker_config.optimization.hand_loss_weight,
            value_entropy_weight=worker_config.optimization.value_entropy_weight,
            reward_entropy_weight=worker_config.optimization.reward_entropy_weight,
            is_terminal_loss_weight=worker_config.optimization.is_terminal_loss_weight,
            optimizer=optimizer,
            min_buffer_size=worker_config.optimization.min_buffer_size,
            updates_per_step=worker_config.optimization.updates_per_step,
            store_model_weights_after=worker_config.optimization.store_model_weights_after,
            store_buffer=worker_config.optimization.store_buffer,
            grad_clip_norm=worker_config.optimization.grad_clip_norm,
            dldl=worker_config.optimization.dldl,
            value_mse=worker_config.optimization.value_mse,
            reward_mse=worker_config.optimization.reward_mse,
            log_gradients=worker_config.optimization.log_gradients,
            log_inputs=worker_config.optimization.log_inputs,
            value_td_5_step=worker_config.optimization.value_td_5_step,
            max_steps_per_second=worker_config.optimization.max_steps_per_second
        )

        if not worker_config.optimization.supervised_targets:
            connector = WorkerConnector(
                model_weights_path=data_path / "latest_network.pd" / "weights.pkl",
                worker_config_path=data_path / "worker_config.json",
                local_game_data_path=data_path / "game_data"
            )
            connector.run(port=worker_config.optimization.port)

        iterations = worker_config.optimization.total_steps // worker_config.optimization.updates_per_step

        logging.info(f"Starting training process for {worker_config.optimization.total_steps} steps over "
                     f"{iterations} iterations with {worker_config.optimization.updates_per_step} updates per step")
        trainer.fit(iterations, Path(network_path))

