from src.operations.training import Training
from src.settings.run_config import ExpConfig


class TrainingRunner:
    def __init__(
        self,
        config: ExpConfig,
    ) -> None:
        self.config = config
        
        # Assets are loaded from factories
        # load previous model if available
        # load previous optimizer if available
        # load previous tracker if available
        # set up training parameters
        self.training: Training

    def run(self) -> None:
        self.training.run()
