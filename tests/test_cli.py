from __future__ import annotations

from src.cli import CLIParser
from src.config import Config


def test_build_config_overrides_many_fields() -> None:
    cli = CLIParser()
    cfg = Config()

    out = cli.build_config(
        cfg,
        args=[
            "--batch_size",
            "64",
            "--lr",
            "0.005",
            "--epochs",
            "3",
            "--num_workers",
            "2",
            "--num_threads",
            "4",
            "--num_interop_threads",
            "2",
            "--hidden_dims",
            "128",
            "32",
            "--dropout",
            "0.1",
            "--emb_max_dim",
            "16",
            "--activation",
            "gelu",
            "--lr_plateau_patience",
            "5",
            "--lr_plateau_factor",
            "0.7",
            "--train_path",
            "./train.csv",
            "--test_path",
            "./test.csv",
            "--artifacts_dir",
            "./out",
            "--log_level",
            "DEBUG",
            "--experiment_name",
            "exp1",
            "--run_name",
            "run1",
            "--resume_from",
            "./out/last.pt",
        ],
    )

    assert out.batch_size == 64
    assert out.lr == 0.005
    assert out.epochs == 3
    assert out.num_workers == 2
    assert out.num_threads == 4
    assert out.num_interop_threads == 2
    assert out.hidden_dims == (128, 32)
    assert out.dropout == 0.1
    assert out.emb_max_dim == 16
    assert out.activation == "gelu"
    assert out.lr_plateau_patience == 5
    assert out.lr_plateau_factor == 0.7
    assert out.train_path == "./train.csv"
    assert out.test_path == "./test.csv"
    assert out.artifacts_dir == "./out"
    assert out.log_level == "DEBUG"
    assert out.experiment_name == "exp1"
    assert out.run_name == "run1"
    assert out.resume_from == "./out/last.pt"


def test_build_config_boolean_optional_flags() -> None:
    cli = CLIParser()
    cfg = Config(deterministic=True, save_last_checkpoint=True)

    out = cli.build_config(
        cfg,
        args=["--no-deterministic", "--no-save_last_checkpoint"],
    )

    assert out.deterministic is False
    assert out.save_last_checkpoint is False
