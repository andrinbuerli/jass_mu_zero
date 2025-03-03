import gc
import json
import logging
import multiprocessing as mp
import sys
import time
from multiprocessing import Process
from pathlib import Path
from threading import Thread

import tqdm
from jass.game.const import team

mp.set_start_method('spawn', force=True)

sys.path.append('../../')

import numpy as np
import itertools

from jass_mu_zero.environment.networking.worker_config import WorkerConfig


def _play_games_(n_games_to_play, agent1_config, agent2_config, network1, network2, queue):
    from jass_mu_zero.factory import get_agent
    agent1 = get_agent(agent1_config, network1,
                       greedy=agent1_config.agent.greedy if hasattr(agent1_config.agent, "greedy") else False)
    agent2 = get_agent(agent2_config, network2,
                       greedy=agent2_config.agent.greedy if hasattr(agent2_config.agent, "greedy") else False)

    rng = range(n_games_to_play)
    for _ in rng:
        from jass_gym.env.jass_single_agent_env import SchieberJassSingleAgentEnv
        from jass_mu_zero.environment.multi_player_game import MultiPlayerGame
        from jass_mu_zero.observation.identity_observation_builder import IdentityObservationBuilder

        game = MultiPlayerGame(env=SchieberJassSingleAgentEnv(observation_builder=IdentityObservationBuilder()))

        first_team = True # np.random.choice([True, False])

        if first_team:
            _, rewards, _, _, _ = game.play_rounds(get_agent=lambda key: {0: agent1, 1: agent2}[team[key]], n=1)
            points = np.array([np.sum(rewards[:, 0]), np.sum(rewards[:, 1])])
            result = (points[0] / points.sum(), points[1] / points.sum())
        else:
            _, rewards, _, _, _ = game.play_rounds(get_agent=lambda key: {1: agent1, 0: agent2}[team[key]], n=1)
            points = np.array([np.sum(rewards[:, 0]), np.sum(rewards[:, 1])])
            result = (points[1] / points.sum(), points[0] / points.sum())

        queue.put(result)

    del agent1, agent2

    queue.put(None)


def _play_games_threaded_(
        n_games,
        parallel_threads_per_evaluation_process,
        agent1_config: WorkerConfig,
        agent2_config: WorkerConfig,
        results_queue):
    from jass_mu_zero.util import set_allow_gpu_memory_growth
    set_allow_gpu_memory_growth(True)

    from jass_mu_zero.factory import get_network
    network1 = get_network(agent1_config, agent1_config.agent.network_path) if hasattr(agent1_config.agent, "network_path") else None
    network2 = get_network(agent2_config, agent2_config.agent.network_path) if hasattr(agent2_config.agent, "network_path") else None

    threads = []
    for k in range(parallel_threads_per_evaluation_process):
        games_to_play_per_thread = (n_games // parallel_threads_per_evaluation_process) + 1
        t = Thread(target=_play_games_, args=(
            games_to_play_per_thread,
            agent1_config,
            agent2_config,
            network1,
            network2,
            results_queue))
        threads.append(t)
        t.start()

    [x.join() for x in threads]


def _evaluate_(
        general_config,
        agent1_config,
        agent2_config,
        skip_on_result_file,
        parallel_processes_per_evaluation,
        parallel_threads_per_evaluation_process,
        result_folder):
    result_file = Path(__file__).parent / "agents_eval_results" / result_folder / f"{agent1_config['note']}-vs-{agent2_config['note']}.json"

    if agent1_config["skip"] and agent2_config["skip"]:
        logging.info(f"skipping flag set for both agents, skipping...")
        return

    if skip_on_result_file and result_file.exists():
        logging.info(f"result file already exists at: {result_file}, skipping...")
        return

    logging.info(f"starting {agent1_config['note']}-vs-{agent2_config['note']} "
                 f"with {parallel_processes_per_evaluation} parallel processes with {parallel_threads_per_evaluation_process} "
                 f"game threads each")

    from jass_mu_zero.factory import get_features

    worker_config1 = WorkerConfig()
    if 'experiment_path' in agent1_config:
        experiment_path = Path(agent1_config['experiment_path'])
        worker_config1.load_from_json(experiment_path / 'worker_config.json')
        worker_config1.network.feature_extractor = get_features(worker_config1.network.feature_extractor)
        worker_config1.agent.__dict__.update(**agent1_config)
        worker_config1.agent.network_path = experiment_path / "latest_network.pd"
    else:
        worker_config1.agent.__dict__ = {**(worker_config1.agent.__dict__), **agent1_config}

    worker_config2 = WorkerConfig()
    if 'experiment_path' in agent2_config:
        experiment_path = Path(agent2_config['experiment_path'])
        worker_config2.load_from_json(experiment_path / 'worker_config.json')
        worker_config2.network.feature_extractor = get_features(worker_config2.network.feature_extractor)
        worker_config2.agent.__dict__.update(**agent2_config)
        worker_config2.agent.network_path = experiment_path / "latest_network.pd"
    else:
        worker_config2.agent.__dict__ = {**(worker_config2.agent.__dict__), **agent2_config}

    queue = mp.Queue()
    processes = []
    total_games = general_config["n_games"]
    games_per_process = total_games // parallel_processes_per_evaluation
    for k in range(parallel_processes_per_evaluation):
        p = Process(target=_play_games_threaded_,
                    args=(
                        games_per_process,
                        parallel_threads_per_evaluation_process,
                        worker_config1,
                        worker_config2,
                        queue))
        processes.append(p)
        p.start()

    tmp_result_file = result_file.parent / "tmp" / result_file.name
    tmp_result_file.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm.tqdm(range(total_games), desc=f"{agent1_config['note']}-vs-{agent2_config['note']}", file=sys.stdout)
    pbar.set_description(f"{agent1_config['note']}-vs-{agent2_config['note']}")
    print("-")  # trigger flushing of output stream

    points_agent1 = []
    points_agent2 = []
    while len(points_agent1) < total_games and len(points_agent2) < total_games:
        for points1, points2 in iter(queue.get, None):
            points_agent1.append(points1)
            points_agent2.append(points2)

            mean_agent1 = np.mean(points_agent1)
            mean_agent2 = np.mean(points_agent2)

            pbar.set_postfix({agent1_config['note']: mean_agent1, agent2_config['note']: mean_agent2})
            pbar.update(1)
            print("-")  # trigger flushing of output stream

            with open(str(tmp_result_file), "w") as f:
                json.dump({
                    f"{agent1_config['note']}-mean": mean_agent1,
                    f"{agent2_config['note']}-mean": mean_agent2,
                    agent1_config['note']: [float(x) for x in points_agent1],
                    agent2_config['note']: [float(x) for x in points_agent2]
                }, f)

    logging.info(
        f"{agent1_config['note']}-vs-{agent2_config['note']}: {np.mean(points_agent1)} - {np.mean(points_agent2)}")

    logging.info(f"{agent1_config['note']}-vs-{agent2_config['note']}: finished {general_config['n_games']}")

    with open(str(result_file), "w") as f:
        json.dump({
            f"{agent1_config['note']}-mean": np.mean(points_agent1),
            f"{agent2_config['note']}-mean": np.mean(points_agent2),
            agent1_config['note']: [float(x) for x in points_agent1],
            agent2_config['note']: [float(x) for x in points_agent2]
        }, f)

    gc.collect()

    logging.info(f"finished {agent1_config['note']}-vs-{agent2_config['note']}")


class MuZeroEvaluationCLI:
    @staticmethod
    def setup_args(parser):
        parser.add_argument(f'--parallel_evaluations', default=1, type=int, help="Number of max parallel evaluations")
        parser.add_argument(f'--parallel_processes_per_evaluation', default=1, type=int, help="Number of max parallel processes per evaluation")
        parser.add_argument(f'--parallel_threads_per_evaluation_process', default=1, type=int, help="Number of max parallel threads per process per evaluation")
        parser.add_argument(f'--no_skip_on_result_file', default=False, action="store_true", help="Skip evaluation if there exists a corresponding result file")
        parser.add_argument(f'--files', nargs="+", default=[None], help="Filenames of evaluations to be executed (relative to folder resources/evaluation)")
        parser.add_argument(f'--all', default=False, action="store_true", help="run all evaluations from resources/evaluation")
        parser.add_argument(f'--folder', default="results", help="Folder to store evaluation results (relative to this script file)")

    @staticmethod
    def run(args):
        base_path = Path(__file__).resolve().parent.parent.parent / "resources" / "evaluation"

        if args.all:
            files = list(base_path.glob("**/*.json"))
        else:
            files = [base_path / file for file in args.files]

        for path in files:
            with open(path, "r") as f:
                config = json.load(f)

            (Path(__file__).resolve().parent / args.folder).mkdir(parents=True, exist_ok=True)

            processes = []
            for comb in list(itertools.combinations(config["agents"], r=2)):
                p = Process(target=_evaluate_, args=(
                    config, *comb,
                    not args.no_skip_on_result_file,
                    args.parallel_processes_per_evaluation,
                    args.parallel_threads_per_evaluation_process,
                    args.folder))
                processes.append(p)
                p.start()

                nr_running_processes = len([x for x in processes if x.is_alive()])
                while 0 < args.parallel_evaluations <= nr_running_processes:
                    time.sleep(1)
                    nr_running_processes = len([x for x in processes if x.is_alive()])

            [x.join() for x in processes]

