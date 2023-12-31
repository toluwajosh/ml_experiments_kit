from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



class FashionMNIST:
    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    def __init__(self, batch_size=64) -> None:
        self._train_dataloader = DataLoader(self.training_data, batch_size=batch_size)
        self._test_dataloader = DataLoader(self.test_data, batch_size=batch_size)
        
        
    @property
    def train_dataloader(self):
        return self._train_dataloader
    
    @property
    def test_dataloader(self):
        return self._test_dataloader