from src.criteria.core import Criterion
from src.datasets.torch_inbuilt import FashionMNIST
from src.models.mlp import MLP
from src.operations.training import Training
from src.optimizers.core import Optimizer
from src.settings.run_config import ExpConfig
from src.tracking.mlflow_tracking import MLFlowTracker

CONFIG = ExpConfig()
model = MLP()
tracker = MLFlowTracker(name="0", params=CONFIG.to_dict())

# Initialize the loss function
loss_fn = Criterion(loss=CONFIG.loss, params=CONFIG.loss_params)
optimizer_module = Optimizer(
    optimizer=CONFIG.optimizer,
    params={"params": model.parameters(), "lr": CONFIG.learning_rate},
    **CONFIG.optimizer_params,
).get_module()

Training(
    config=CONFIG,
    model=model,
    criterion=loss_fn,
    optimizer=optimizer_module,
    tracker=tracker,
    dataset=FashionMNIST(batch_size=CONFIG.batch_size),
).run()
