from dataclasses import dataclass
from dataclasses import asdict
from typing import Optional
from datetime import datetime

@dataclass
class Config:
    # data / training
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    seed: int = 42
    num_workers: int = 4
    grad_clip: float = 1.0
    early_stopping_patience: int = 7
    test_size: float = 0.2

    # cpu runtime
    num_threads: int | None = None  # None -> auto
    num_interop_threads: int | None = 1

    # model
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.3
    emb_max_dim: int = 32
    activation: str = "relu"

    # scheduler
    lr_plateau_patience: int = 2
    lr_plateau_factor: float = 0.5

    # misc
    train_path: str = "./data/dataset_train.csv"
    test_path: str = "./data/dataset_test.csv"
    artifacts_dir: str = "artifacts"
    log_level: str = "INFO"
    run_name: Optional[str] = None
    experiment_name: str = "click_model"
    deterministic: bool = True
    save_last_checkpoint: bool = True
    resume_from: Optional[str] = None

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self):
        return asdict(self)

    def update_from_dict(self, d: dict):
        for k, v in d.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)

def get_default_config() -> Config:
    return Config()