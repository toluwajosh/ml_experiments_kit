import mlflow
import torch

from src.criteria.core import Criterion
from src.datasets.torch_inbuilt import FashionMNIST
from src.models.mlp import MLP
from src.optimizers.core import Optimizer

mlflow.autolog()

train_dataloader = FashionMNIST().train_dataloader
test_dataloader = FashionMNIST().test_dataloader

model = MLP()

learning_rate = 1e-3
batch_size = 64
epochs = 5
loss = "torch.nn.CrossEntropyLoss"
loss_params = {}
optimizer = "torch.optim.SGD"


# Initialize the loss function
loss_fn = Criterion(loss=loss, params=loss_params)
optimizer_module = Optimizer(
    optimizer=optimizer,
    params={"params": model.parameters(), "lr": learning_rate},
).get_module()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer_module)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
