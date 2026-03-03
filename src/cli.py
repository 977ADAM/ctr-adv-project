from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Sequence

from .config import Config


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
        # data / training
        self.parser.add_argument("--batch_size", type=int, default=None)
        self.parser.add_argument("--lr", type=float, default=None)
        self.parser.add_argument("--weight_decay", type=float, default=None)
        self.parser.add_argument("--epochs", type=int, default=None)
        self.parser.add_argument("--seed", type=int, default=None)
        self.parser.add_argument("--num_workers", type=int, default=None)
        self.parser.add_argument("--grad_clip", type=float, default=None)
        self.parser.add_argument("--early_stopping_patience", type=int, default=None)
        self.parser.add_argument("--test_size", type=float, default=None)

        # cpu runtime
        self.parser.add_argument(
            "--num_threads",
            type=int,
            default=None,
            help="Torch intra-op threads (default: auto from config)",
        )
        self.parser.add_argument(
            "--num_interop_threads",
            type=int,
            default=None,
        )

        # model
        self.parser.add_argument(
            "--hidden_dims",
            type=int,
            nargs="+",
            default=None,
            help="Hidden layer sizes, e.g. --hidden_dims 256 128",
        )
        self.parser.add_argument("--dropout", type=float, default=None)
        self.parser.add_argument("--emb_max_dim", type=int, default=None)
        self.parser.add_argument(
            "--activation",
            type=str,
            choices=["relu", "gelu", "silu"],
            default=None,
        )

        # scheduler
        self.parser.add_argument("--lr_plateau_patience", type=int, default=None)
        self.parser.add_argument("--lr_plateau_factor", type=float, default=None)

        # misc / paths
        self.parser.add_argument(
            "--train_path",
            type=str,
            default=None,
            help="Path to train CSV file",
        )

        self.parser.add_argument(
            "--test_path",
            type=str,
            default=None,
            help="Path to test CSV file",
        )

        self.parser.add_argument(
            "--artifacts_dir",
            type=str,
            default=None,
            help="Directory for experiment artifacts",
        )

        self.parser.add_argument(
            "--log_level",
            type=str,
            default=None,
            help="Logging level",
        )
        self.parser.add_argument("--run_name", type=str, default=None)
        self.parser.add_argument("--experiment_name", type=str, default=None)
        self.parser.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        self.parser.add_argument(
            "--save_last_checkpoint",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        self.parser.add_argument("--resume_from", type=str, default=None)

    def parse(self, args: Sequence[str] | None = None) -> argparse.Namespace:
        return self.parser.parse_args(args)

    def build_config(
        self, base_config: Config, args: Sequence[str] | None = None
    ) -> Config:
        """
        Merge CLI args into Config in a controlled way.
        """
        parsed_args = self.parse(args)
        config_dict = asdict(base_config)

        for key, value in vars(parsed_args).items():
            if key in config_dict and value is not None:
                if key == "hidden_dims":
                    value = tuple(value)
                setattr(base_config, key, value)

        return base_config
