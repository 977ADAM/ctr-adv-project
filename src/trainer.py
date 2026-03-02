import torch
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        stopper,
        device,
        config,
        logger,
        artifacts_dir: Path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.stopper = stopper
        self.device = device
        self.config = config
        self.logger = logger
        self.artifacts_dir = artifacts_dir
        self.start_epoch = 1
        self.best_auc = -1.0

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_auc = ckpt.get("best_auc", -1.0)
        self.logger.info(f"Resumed from {path}")

    def fit(self, train_loader, val_loader):
        best_auc = self.best_auc
        history = []

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            tr_loss = self._train_epoch(train_loader)
            va_loss, va_auc = self._evaluate(val_loader)

            self.scheduler.step(va_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_auc": va_auc,
                "lr": current_lr,
            }

            history.append(row)
            self.logger.info(json.dumps(row))

            if va_auc > best_auc:
                best_auc = va_auc
                self._save_checkpoint("best.pt", epoch, best_auc)

            if self.config.save_last_checkpoint:
                self._save_checkpoint("last.pt", epoch, best_auc)

            if self.stopper.step(va_auc):
                break

        return best_auc, history
    
    def _save_checkpoint(self, filename, epoch, best_auc):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_auc": best_auc,
                "config": self.config.to_dict(),
            },
            self.artifacts_dir / filename,
        )

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for X_num, X_cat, y_batch in loader:
            X_num = X_num.to(self.device)
            X_cat = X_cat.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model((X_num, X_cat))
            loss = self.criterion(logits, y_batch)
            loss.backward()

            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()
            total_loss += float(loss.item())

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def _evaluate(self, loader):

        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []

        for X_num, X_cat, y_batch in loader:
            X_num = X_num.to(self.device)
            X_cat = X_cat.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model((X_num, X_cat))
            loss = self.criterion(logits, y_batch)
            total_loss += float(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_targets.append(
                y_batch.cpu().numpy().reshape(-1)
            )

        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        auc = roc_auc_score(all_targets, all_probs)

        return total_loss / max(1, len(loader)), float(auc)