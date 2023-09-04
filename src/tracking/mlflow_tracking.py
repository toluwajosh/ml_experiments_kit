from typing import Dict, Optional

import mlflow


class MLFlowTracker:
    def __init__(
        self, name: Optional[str] = None, params: Optional[Dict] = None
    ):
        self.params = params
        mlflow.start_run(experiment_id=name)
        print(mlflow.get_tracking_uri())

        self.log_params()

    def log_params(self):
        mlflow.log_params(self.params)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_artifacts(self, local_dir, artifact_path=None):
        mlflow.log_artifacts(local_dir, artifact_path)
