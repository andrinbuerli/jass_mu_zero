import logging
import multiprocessing as mp

from jass_mu_zero.agent.agent_from_cpp import AgentFromCpp
from jass_mu_zero.agent.agent_from_cpp_cheating import AgentFromCppCheating
from jass_mu_zero.environment.networking.worker_config import WorkerConfig
from jass_mu_zero.environment.service.player_service_app import PlayerServiceApp
from jass_mu_zero.factory import get_agent, get_network

mp.set_start_method('spawn', force=True)
from multiprocessing import Process
from pathlib import Path


def host_agent(config: WorkerConfig):
    try:
        network = get_network(config, config.network.path)
    except NotImplementedError:
        network = None

    agent = get_agent(config, network, force_local=True)
    if config.agent.cheating:
        agent = AgentFromCppCheating(agent=agent)
    else:
        agent = AgentFromCpp(agent=agent)
    app = PlayerServiceApp("jass_agents")
    name = config.agent.name if config.agent.name is not None else config.agent.type
    logging.info(f"Hosting player {config.agent.port}/{name}")
    app.add_player(name, agent)
    app.run(host="0.0.0.0", port=config.agent.port)


class HostAgentsCLI:

    @staticmethod
    def setup_args(parser):
        parser.add_argument(f'--files', nargs='+', default=[])

    @staticmethod
    def run(args):
        processes = []
        base_path = Path(__file__).resolve().parent.parent.parent / "resources" / "baselines"

        for agent_str in args.files:
            config = WorkerConfig()
            config.load_from_json(base_path / agent_str)
            p = Process(target=host_agent, args=[config])
            p.start()

        [p.join() for p in processes]