import argparse
import logging
import os
import sys
from pathlib import Path

from typing import Callable


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

        parser.add_argument("--attach", help="Attach to experiment docker container", action="store_true")
        parser.add_argument("--baselines", help="Start hosting baselines container", action="store_true")
        parser.add_argument("--nodocker", help="Run experiment without docker", action="store_true")

        subparsers = parser.add_subparsers(dest="command")

        eval_parser = subparsers.add_parser("train", description="Train MuZero model")
        MuZeroTrainingCLI.setup_arguments(eval_parser)

        eval_parser = subparsers.add_parser("collect", description="Sample data for MuZero Training")
        MuZeroDataCollectionCLI.setup_arguments(eval_parser)

        eval_parser = subparsers.add_parser("eval", description="Run evaluation")
        MuZeroEvaluationCLI.setup_arguments(eval_parser)

        self.args, self.unknown_args = parser.parse_known_args()

    def __call__(self):
        if self.args.command == "train" and self.args.nodocker:
            MuZeroTrainingCLI().run(self.args)
            return

        if self.args.command == "collect" and self.args.nodocker:
            MuZeroDataCollectionCLI().run(self.args)
            return

        if self.args.command == "eval" and self.args.nodocker:
            MuZeroEvaluationCLI().run(self.args)
            return

        if self.args.baselines:
            command = f"docker-compose -f resources " \
                      f"{Path(__file__).parent.parent.parent / 'resources' / 'baselines' / 'baselines.yml'} up -d"
        else:
            command = 'bash -c "'
            command += 'sjmz --nodocker ' + " ".join(sys.argv[1:])
            command += '"'

            attach = (self.args.attach and not self.args.nodocker)
            command = "docker-compose run " + ("" if attach else " -d ") + "mam_gym " + command

        print("executing", command, "in docker container")

        os.system(command)

if __name__ == "__main__":
    cli = SchieberJassMuZeroCli()
    cli()

