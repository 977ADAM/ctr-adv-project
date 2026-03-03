import os
import random
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import get_default_config
from .logging_utils import setup_logging
from .trainer import Trainer
from .cli import CLIParser
from .preprocessing import CTRPreprocessor
from .inference import CTRInferenceService
from .model import (
    ClickModel,
    ClickDataset,
    EarlyStopping,
    make_loader,
)


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)


def get_device() -> torch.device:
    return torch.device("cpu")


def setup_runtime(config) -> None:
    set_seed(config.seed, config.deterministic)

    if config.num_threads is None:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    else:
        torch.set_num_threads(max(1, int(config.num_threads)))
    if config.num_interop_threads is not None:
        torch.set_num_interop_threads(max(1, int(config.num_interop_threads)))


def _drop_session_id(df: pd.DataFrame) -> pd.DataFrame:
    if "session_id" in df.columns:
        return df.drop(columns=["session_id"])
    return df


def _load_datasets(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return _drop_session_id(df_train), _drop_session_id(df_test)


def _prepare_training_data(
    df: pd.DataFrame, config: Any
) -> tuple[CTRPreprocessor, Any, Any, np.ndarray, int, list[int]]:
    preprocessor = CTRPreprocessor()
    x_train, x_val, y_train, y_val = preprocessor.make_splits(
        df,
        test_size=config.test_size,
    )

    x_train_num, x_train_cat = preprocessor.fit_transform(x_train)
    x_val_num, x_val_cat = preprocessor.transform(x_val)

    train_dataset = ClickDataset(x_train_num, x_train_cat, y_train)
    val_dataset = ClickDataset(x_val_num, x_val_cat, y_val)
    train_loader = make_loader(train_dataset, config.batch_size, True, config.num_workers)
    val_loader = make_loader(val_dataset, config.batch_size, False, config.num_workers)

    cat_cardinalities = [
        int(x_train_cat[:, col_i].max()) + 1
        for col_i in range(len(preprocessor.categorical_cols))
    ]
    num_features = x_train_num.shape[1]
    return preprocessor, train_loader, val_loader, y_train, num_features, cat_cardinalities


def _build_training_components(
    config: Any,
    device: torch.device,
    num_features: int,
    cat_cardinalities: list[int],
    y_train: np.ndarray,
) -> tuple[Any, Any, Any, Any, Any]:
    model = ClickModel(
        num_features=num_features,
        cat_cardinalities=cat_cardinalities,
        config=config,
    ).to(device)

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config.lr_plateau_patience,
        factor=config.lr_plateau_factor,
    )
    stopper = EarlyStopping(patience=config.early_stopping_patience, mode="max")
    return model, criterion, optimizer, scheduler, stopper


def _save_artifacts(
    artifacts_dir: Path,
    preprocessor: CTRPreprocessor,
    cat_cardinalities: list[int],
    best_auc: float,
    history: list[dict[str, Any]],
    config: Any,
) -> None:
    preprocessor.save(artifacts_dir)
    meta = {
        "categorical_cols": preprocessor.categorical_cols,
        "numerical_cols": preprocessor.numerical_cols,
        "cat_cardinalities": cat_cardinalities,
        "best_val_auc": best_auc,
        "config": config.to_dict(),
        "feature_schema_version": 1,
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (artifacts_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))


def _run_inference(
    artifacts_dir: Path,
    df_test: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Path:
    inference_service = CTRInferenceService(
        artifacts_dir=artifacts_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    sub_path = artifacts_dir / "test_predictions.csv"
    inference_service.predict_to_csv(df_test, sub_path)
    return sub_path


def main():
    cli = CLIParser()
    config = cli.build_config(get_default_config())

    setup_runtime(config)
    artifacts_dir = Path(config.artifacts_dir) / config.experiment_name / config.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(config.log_level, str(artifacts_dir / "train.log"))
    logger.info(f"Run: {config.run_name}")
    logger.info("config", extra={"event": {"config": config.to_dict()}})

    df_train, df_test = _load_datasets(config.train_path, config.test_path)
    preprocessor, train_loader, val_loader, y_train, num_features, cat_cardinalities = (
        _prepare_training_data(df_train, config)
    )

    device = get_device()
    model, criterion, optimizer, scheduler, stopper = _build_training_components(
        config=config,
        device=device,
        num_features=num_features,
        cat_cardinalities=cat_cardinalities,
        y_train=y_train,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        stopper=stopper,
        device=device,
        config=config,
        logger=logger,
        artifacts_dir=artifacts_dir,
    )
    best_auc, history = trainer.fit(train_loader, val_loader)
    _save_artifacts(
        artifacts_dir=artifacts_dir,
        preprocessor=preprocessor,
        cat_cardinalities=cat_cardinalities,
        best_auc=best_auc,
        history=history,
        config=config,
    )
    sub_path = _run_inference(
        artifacts_dir=artifacts_dir,
        df_test=df_test,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    logger.info(f"Saved: {sub_path}")


if __name__ == "__main__":
    main()
