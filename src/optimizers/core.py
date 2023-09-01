import importlib
from typing import Dict, Optional, Type

import torch


class Optimizer:
    def __init__(self, optimizer: str, params: Dict) -> None:
        try:
            self._optimizer: torch.optim.Optimizer = self.obtain_class(
                optimizer
            )
            self._optimizer = self._optimizer(**params)
        except ModuleNotFoundError as error:
            raise error

    @staticmethod
    def obtain_class(path: str) -> Type:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        attribute = getattr(module, class_name)
        return attribute

    def get_module(self):
        return self._optimizer
