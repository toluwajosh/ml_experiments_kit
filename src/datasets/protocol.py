from typing import Protocol


class Dataset(Protocol):
    @property
    def train_dataloader(self):
        pass

    @property
    def test_dataloader(self):
        pass
