import argparse
import logging
import os

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

        parser.add_argument("--train", help="start training from file", type=str)
        parser.add_argument("--collect", help="start data collection using given compose file", type=str)
        parser.add_argument("--eval", help="run evaluation given by string",  nargs="+", default=[], type=list)
        self.args, self.unknown_args = parser.parse_known_args()

    def __call__(self):
        pass


def main():
    cli = SchieberJassMuZeroCli()
    cli()


if __name__ == "__main__":
    main()
