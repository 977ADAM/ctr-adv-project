from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from preprocessing import CTRPreprocessor
from model import ClickModel, ClickDataset, make_loader
from config import Config


class CTRInferenceService:
    """
    Production-style inference service.

    Responsible for:
    - loading model + preprocessing artifacts
    - preparing input data
    - running forward pass
    - returning probabilities
    """

    def __init__(
        self,
        artifacts_dir: Path,
        device: Optional[torch.device] = None,
        batch_size: int = 1024,
        num_workers: int = 0,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.preprocessor = CTRPreprocessor()
        self.model: Optional[ClickModel] = None
        self.meta: Optional[dict] = None

        self._load_artifacts()

    # --------------------- Loading ---------------------

    def _load_artifacts(self):
        self.preprocessor.load(self.artifacts_dir)

        self.meta = json.loads(
            (self.artifacts_dir / "meta.json").read_text()
        )

        self.model = ClickModel(
            num_features=len(self.meta["numerical_cols"]),
            cat_cardinalities=self.meta["cat_cardinalities"],
            config=self._build_inference_config(),
        ).to(self.device)

        ckpt = torch.load(
            self.artifacts_dir / "best.pt",
            map_location=self.device,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _build_inference_config(self):
        """
        Lightweight config reconstruction.
        We only need model hyperparams.
        """
        return Config(**self.meta["config"])

    # --------------------- Public API ---------------------

    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X_num, X_cat = self.preprocessor.transform(df)

        dummy_y = np.zeros(len(X_num), dtype=np.float32)
        ds = ClickDataset(X_num, X_cat, dummy_y)
        loader = make_loader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        probs = []

        for X_num_b, X_cat_b, _ in loader:
            X_num_b = X_num_b.to(self.device)
            X_cat_b = X_cat_b.to(self.device)

            logits = self.model((X_num_b, X_cat_b))
            p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(p)

        return np.concatenate(probs)

    def predict_to_csv(
        self,
        df: pd.DataFrame,
        output_path: Path,
    ):
        probs = self.predict_proba(df)
        sub = pd.DataFrame({"click_proba": probs})
        sub.to_csv(output_path, index=False)