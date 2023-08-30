from models.mlp import MLP
from datasets.torch_inbuilt import FashionMNIST

train_dataloader = FashionMNIST().train_dataloader
test_dataloader = FashionMNIST().test_dataloader

model = MLP()