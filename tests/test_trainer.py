from __future__ import annotations

import logging

import numpy as np
import torch

from src.config import Config
from src.model import ClickDataset, ClickModel, EarlyStopping, make_loader
from src.trainer import Trainer


def _make_loader() -> torch.utils.data.DataLoader:
    x_num = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
        dtype=np.float32,
    )
    x_cat = np.array([[1, 2], [2, 1], [1, 1], [2, 2]], dtype=np.int64)
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    ds = ClickDataset(x_num, x_cat, y)
    return make_loader(ds, batch_size=2, shuffle=False, num_workers=0)


def test_trainer_resume_from_checkpoint(tmp_path) -> None:
    resume_path = tmp_path / "resume.pt"
    cfg = Config(
        hidden_dims=(4,),
        dropout=0.0,
        emb_max_dim=4,
        epochs=3,
        save_last_checkpoint=False,
        resume_from=str(resume_path),
    )
    model = ClickModel(num_features=2, cat_cardinalities=[3, 3], config=cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")

    torch.save(
        {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": 0.77,
            "config": cfg.to_dict(),
        },
        resume_path,
    )

    logger = logging.getLogger("test_trainer")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=torch.nn.BCEWithLogitsLoss(),
        stopper=EarlyStopping(patience=10, mode="max"),
        device=torch.device("cpu"),
        config=cfg,
        logger=logger,
        artifacts_dir=tmp_path,
    )

    assert trainer.start_epoch == 3
    assert trainer.best_auc == 0.77

    loader = _make_loader()
    best_auc, history = trainer.fit(loader, loader)

    assert len(history) == 1
    assert isinstance(best_auc, float)
    row = history[0]
    assert "val_auc" in row
    assert "val_pr_auc" in row
    assert "val_logloss" in row
    assert "val_brier" in row
    assert "val_ece" in row
