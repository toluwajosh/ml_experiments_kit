import unittest

import torch

from src.criteria.core import Criterion


class TestCriterion(unittest.TestCase):
    def test_valid_loss(self):
        # Test if the class can correctly load a valid loss function and calculate loss
        params = {"reduction": "none"}
        criterion = Criterion("torch.nn.MSELoss", params)

        logits = torch.tensor([1.0, 2.0, 3.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0])

        loss = criterion(logits, ground_truth)
        self.assertEqual(
            loss.item(), 0.0
        )  # Expected loss should be zero as logits are same as ground_truth

    def test_invalid_loss(self):
        # Test if the class raises an error when given an invalid loss function
        params = {"reduction": "none"}

        with self.assertRaises(Exception) as context:
            Criterion("torch.nn.InvalidLoss", params)

        self.assertTrue(
            "'torch.nn' has no attribute 'InvalidLoss'"
            in str(context.exception)
        )

    def test_with_weights(self):
        # Test if the class can correctly use weights to calculate the loss
        params = {"reduction": "none"}
        criterion = Criterion("torch.nn.MSELoss", params)

        logits = torch.tensor([1.0, 2.0, 3.0])
        ground_truth = torch.tensor([1.5, 2.5, 3.5])
        weights = torch.tensor([0.1, 0.2, 0.3])

        loss = criterion(logits, ground_truth, weights)
        expected_loss = (
            (logits - ground_truth) ** 2 * weights
        ).sum() / weights.sum()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=6)
