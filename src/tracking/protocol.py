from typing import Protocol


class Tracker(Protocol):
    def log_params(self):
        pass

    def log_param(self, key, value):
        pass

    def log_metric(self, key, value):
        pass

    def log_artifacts(self, local_dir, artifact_path=None):
        pass
