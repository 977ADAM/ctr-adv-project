from __future__ import annotations

from src.config import Config


def test_config_generates_run_name_when_missing() -> None:
    cfg = Config(run_name=None)
    assert cfg.run_name is not None
    assert len(cfg.run_name) == 15


def test_update_from_dict_updates_known_and_non_none_fields() -> None:
    cfg = Config(batch_size=32, activation="relu")
    cfg.update_from_dict({"batch_size": 64, "activation": None, "missing": 1})

    assert cfg.batch_size == 64
    assert cfg.activation == "relu"
