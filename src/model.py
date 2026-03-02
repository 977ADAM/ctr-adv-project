from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score


# ================= DATASET =================

class ClickDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )


# ================= MODEL =================

class ClickModel(nn.Module):
    def __init__(self, num_features: int, cat_cardinalities: List[int], config):
        super().__init__()

        def emb_dim(c):
            return min(config.emb_max_dim, int(1.6 * (c ** 0.56)))

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim(cardinality), padding_idx=0)
            for cardinality in cat_cardinalities
        ])

        emb_dim_total = sum([emb_dim(c) for c in cat_cardinalities])
        input_dim = num_features + emb_dim_total

        layers = []
        prev_dim = input_dim

        def get_activation(name: str):
            name = name.lower()
            if name == "relu":
                return nn.ReLU()
            if name == "gelu":
                return nn.GELU()
            if name == "silu":
                return nn.SiLU()
            raise ValueError(f"Unknown activation: {name}")

        for h in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                get_activation(config.activation),
                nn.Dropout(config.dropout),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x_num, x_cat = x

        emb = [
            emb_layer(x_cat[:, i])
            for i, emb_layer in enumerate(self.embeddings)
        ]
        emb = torch.cat(emb, dim=1)

        x = torch.cat([x_num, emb], dim=1)
        return self.mlp(x)


# ================= EARLY STOPPING =================

class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best
            if self.mode == "max"
            else value < self.best
        )

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# ================= EVALUATION =================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    for X_num, X_cat, y_batch in loader:
        X_num = X_num.to(device)
        X_cat = X_cat.to(device)
        y_batch = y_batch.to(device)

        logits = model((X_num, X_cat))
        loss = criterion(logits, y_batch)
        total_loss += float(loss.item())

        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        all_probs.append(probs)
        all_targets.append(y_batch.cpu().numpy().reshape(-1))

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    auc = roc_auc_score(all_targets, all_probs)

    return total_loss / max(1, len(loader)), float(auc)
