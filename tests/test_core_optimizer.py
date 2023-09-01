import unittest

import torch

from src.models.mlp import MLP
from src.optimizers.core import Optimizer


class TestOptimizer(unittest.TestCase):
    model = MLP()

    def test_valid_optimizer(self):
        # Test if the class can correctly load a valid optimizer and initialize it
        params = {
            "params": self.model.parameters(),
            "lr": 0.01,
        }
        optimizer = Optimizer("torch.optim.SGD", params)

        self.assertIsInstance(optimizer.get_module(), torch.optim.SGD)

    def test_invalid_optimizer(self):
        # Test if the class raises an error when given an invalid optimizer
        params = {
            "params": self.model.parameters(),
            "lr": 0.01,
        }

        with self.assertRaises(Exception) as context:
            Optimizer("torch.optim.InvalidOptimizer", params)

        self.assertTrue(
            "'torch.optim' has no attribute 'InvalidOptimizer'"
            in str(context.exception)
        )
