import importlib
from typing import Dict, Optional, Type

import torch


class Criterion(torch.nn.Module):
    def __init__(self, loss: str, params: Dict) -> None:
        super(Criterion, self).__init__()
        try:
            self._loss: torch.nn.modules.loss._Loss = self.obtain_class(loss)
            self._loss = self._loss(**params)
        except ModuleNotFoundError as error:
            raise error

        self._loss.reduction = "none"

    @staticmethod
    def obtain_class(path: str) -> Type:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        attribute = getattr(module, class_name)
        return attribute

    def forward(
        self,
        logits: torch.Tensor,
        ground_truth: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self._loss(logits, ground_truth)

        if weights is not None:
            loss *= weights
            loss = loss.sum(dim=0) / (weights.sum(dim=0) + 1e-8)

        loss = loss.mean()
        return loss
