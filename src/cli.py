from __future__ import annotations

import argparse
from dataclasses import asdict

from config import Config


class CLIParser:
    """
    Responsible only for:
    - defining CLI arguments
    - parsing them
    - merging with Config
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Click-through rate model training (CPU-only)"
        )
        self._register_arguments()

    def _register_arguments(self) -> None:
        # paths
        self.parser.add_argument(
            "--train_path",
            type=str,
            default="./data/dataset_train.csv",
            help="Path to train CSV file",
        )

        self.parser.add_argument(
            "--test_path",
            type=str,
            default="./data/dataset_test.csv",
            help="Path to test CSV file",
        )

        self.parser.add_argument(
            "--artifacts_dir",
            type=str,
            default="artifacts",
            help="Directory for experiment artifacts",
        )

        self.parser.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            help="Logging level",
        )

    def parse(self) -> argparse.Namespace:
        return self.parser.parse_args()

    def build_config(self, base_config: Config) -> Config:
        """
        Merge CLI args into Config in a controlled way.
        """
        args = self.parse()
        config_dict = asdict(base_config)

        for key, value in vars(args).items():
            if key in config_dict and value is not None:
                setattr(base_config, key, value)

        return base_config