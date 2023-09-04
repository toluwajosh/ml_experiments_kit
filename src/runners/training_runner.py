from typing import Optional

import torch

from src.datasets.protocol import Dataset
from src.settings.run_config import ExpConfig
from src.tracking.protocol import Tracker


class TrainingRunner:
    def __init__(
        self,
        config: ExpConfig,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        tracker: Tracker,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset],
    ) -> None:
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.tracker = tracker
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def run(self) -> None:
        pass
