import os
import logging
import pandas as pd
import numpy as np
import random
from pathlib import Path
import json
import torch
from torch import nn

try:
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
    
except ImportError:
    from config import get_default_config
    from logging_utils import setup_logging
    from trainer import Trainer
    from cli import CLIParser
    from preprocessing import CTRPreprocessor
    from inference import CTRInferenceService
    from model import (
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
    # строго CPU (по ТЗ)
    return torch.device("cpu")

def setup_runtime(config) -> None:
    set_seed(config.seed, config.deterministic)

    if config.num_threads is None:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    else:
        torch.set_num_threads(max(1, int(config.num_threads)))
    if config.num_interop_threads is not None:
        torch.set_num_interop_threads(max(1, int(config.num_interop_threads)))

# --------------- Train / Validation split (БЕЗ утечки + стратификация) ------------------

def main():
    cli = CLIParser()
    config = cli.build_config(get_default_config())

    setup_runtime(config)
    artifacts_dir = Path(config.artifacts_dir) / config.experiment_name / config.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(config.log_level, str(artifacts_dir / "train.log"))
    logger.info(f"Run: {config.run_name}")
    logger.info(f"Config: {json.dumps(config.to_dict(), ensure_ascii=False)}")

    df = pd.read_csv(config.train_path)
    df_test = pd.read_csv(config.test_path)

    # базовая чистка
    if "session_id" in df.columns:
        df = df.drop(columns=["session_id"])
    if "session_id" in df_test.columns:
        df_test = df_test.drop(columns=["session_id"])

    preprocessor = CTRPreprocessor()

    X_train, X_val, y_train, y_val = preprocessor.make_splits(
        df,
        test_size=config.test_size,
        seed=config.seed,
    )

    X_train_num, X_train_cat = preprocessor.fit_transform(X_train)
    X_val_num, X_val_cat = preprocessor.transform(X_val)

    train_dataset = ClickDataset(X_train_num, X_train_cat, y_train)
    val_dataset = ClickDataset(X_val_num, X_val_cat, y_val)
    train_loader = make_loader(train_dataset, config.batch_size, True, config.num_workers)
    val_loader = make_loader(val_dataset, config.batch_size, False, config.num_workers)

    # FIX: categorical_cols (раньше было categorical_col и код падал)
    cat_cardinalities = []
    for col_i, col in enumerate(preprocessor.categorical_cols):
        # encoder выдаёт [-1..], после +1 получаем [0..]
        # определяем максимум по train, +1 чтобы вместить max_id
        max_id = int(X_train_cat[:, col_i].max())
        cat_cardinalities.append(max_id + 1)

    device = get_device()
    model = ClickModel(
        num_features=X_train_num.shape[1],
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

    # save preprocessors + metadata
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

    # -------- Inference via service (production-style) --------
    inference_service = CTRInferenceService(
        artifacts_dir=artifacts_dir,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    sub_path = artifacts_dir / "test_predictions.csv"
    inference_service.predict_to_csv(df_test, sub_path)
    logger.info(f"Saved: {sub_path}")

if __name__ == "__main__":
    main()