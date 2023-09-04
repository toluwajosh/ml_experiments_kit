import torch
import numpy as np

from src.datasets.protocol import Dataset
from src.settings.run_config import ExpConfig
from src.tracking.protocol import Tracker


class Training:
    def __init__(
        self,
        config: ExpConfig,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        tracker: Tracker,
        dataset: Dataset,
    ) -> None:
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.tracker = tracker
        self.dataset = dataset

    def run(self) -> None:
        for t in range(self.config.epochs):
            self.tracker.log_metric("epoch", t)
            print(f"Epoch {t+1}\n-------------------------------")
            epoch_loss = self.train_epoch()
            eval_loss, accuracy = self.test()
            self.tracker.log_metric("train_loss", epoch_loss)
            self.tracker.log_metric("eval_loss", eval_loss)
            self.tracker.log_metric("accuracy", accuracy)

    def train_epoch(self):
        size = len(self.dataset.train_dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        losses = []
        for batch, (X, y) in enumerate(self.dataset.train_dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.criterion(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                losses.append(loss)
        return np.mean(losses)

    def test(self):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()
        size = len(self.dataset.test_dataloader.dataset)
        num_batches = len(self.dataset.test_dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in self.dataset.test_dataloader:
                pred = self.model(X)
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        return test_loss, correct * 100
