import argparse
import logging
import os
import sys
from pathlib import Path

from typing import Callable


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SchieberJassMuZeroCli(Callable):
    """
    Entrypoint for the sjmz CLI
    """

    def __init__(self):
        logging.basicConfig(level=logging.ERROR)
        this_dir, this_filename = os.path.split(__file__)
        package_path = os.path.join(this_dir, "..")
        os.chdir(package_path)

        parser = argparse.ArgumentParser(description="Schieber Jass MuZero CLI")

        parser.add_argument("--test", help="run tests", action="store_true")
        parser.add_argument("--attach", help="Attach to experiment docker container", action="store_true")
        parser.add_argument("--baselines", help="Start hosting baselines container", action="store_true")
        parser.add_argument("--nodocker", help="Run experiment without docker", action="store_true")

        subparsers = parser.add_subparsers(dest="command")

        train_parser = subparsers.add_parser("train", description="Train MuZero model")
        try:
            from jass_mu_zero.scripts.train_mu_zero import MuZeroTrainingCLI
            MuZeroTrainingCLI.setup_args(train_parser)
        except:
            logging.error(f"could not setup training cli")

        collect_parser = subparsers.add_parser("collect", description="Sample data for MuZero Training")
        collect_parser.add_argument("--machine", help="Machine name for data collection", default="")
        try:
            from jass_mu_zero.scripts.collect_n_send_game_data import MuZeroDataCollectionCLI
            MuZeroDataCollectionCLI.setup_args(collect_parser)
        except:
            logging.error(f"could not setup data collection cli")

        eval_parser = subparsers.add_parser("eval", description="Run evaluation")
        try:
            from jass_mu_zero.scripts.evaluate_agents import MuZeroEvaluationCLI
            MuZeroEvaluationCLI.setup_args(eval_parser)
        except:
            logging.error(f"could not setup evaluation cli")

        eval_parser = subparsers.add_parser("host", description="Host agents")
        try:
            from jass_mu_zero.scripts.host_agents import HostAgentsCLI
            HostAgentsCLI.setup_args(eval_parser)
        except:
            logging.error(f"could not setup host agents cli")

        self.args, self.unknown_args = parser.parse_known_args()

    def __call__(self):
        if self.args.command == "train" and self.args.nodocker:
            from jass_mu_zero.scripts.train_mu_zero import MuZeroTrainingCLI
            MuZeroTrainingCLI().run(self.args)
            return

        if self.args.command == "eval" and self.args.nodocker:
            from jass_mu_zero.scripts.evaluate_agents import MuZeroEvaluationCLI
            MuZeroEvaluationCLI().run(self.args)
            return

        if self.args.command == "host" and self.args.nodocker:
            from jass_mu_zero.scripts.host_agents import HostAgentsCLI
            HostAgentsCLI().run(self.args)
            return

        command = None

        attach = (self.args.attach and not self.args.nodocker)

        if self.args.test and self.args.nodocker:
            path = Path(__file__).parent.parent / 'test'
            logging.info(f"Running test at {path}")
            os.system(f"pytest --forked -n auto --timeout=120 {path} ")
            return
        elif self.args.test and not self.args.nodocker:
            command = "docker-compose run " + ("" if attach else " -d ")\
                      + "mu_zero pytest  --forked -n auto --timeout=120  -v /app/test"

        if self.args.command == "collect" and self.args.nodocker:
            from jass_mu_zero.scripts.collect_n_send_game_data import MuZeroDataCollectionCLI
            MuZeroDataCollectionCLI().run(self.args)
            return
        elif self.args.command == "collect" and not self.args.nodocker:
            print("machine", self.args.machine )
            file = Path(__file__).parent.parent / 'resources' / 'data_collectors' / (self.args.machine + '.yml')
            command = f"docker-compose -f {file} up" + ("" if attach else " -d ")

        if self.args.baselines and command is None:
            logging.info("Creating jass_net docker network")
            os.system("docker network create jass_net")
            command = f"docker-compose -f " \
                      f"{Path(__file__).parent.parent / 'resources' / 'baselines' / 'baselines.yml'} up " + ("" if attach else " -d ")
        elif command is None:
            command = 'bash -c "'
            command += 'sjmz --nodocker ' + " ".join(sys.argv[1:])
            command += '"'

            command = "docker-compose run " + ("" if attach else " -d ") + "mu_zero " + command

        print("executing", command, "in docker container, attach:", attach)

        os.system(command)


def main():
    cli = SchieberJassMuZeroCli()
    cli()


if __name__ == "__main__":
    main()

