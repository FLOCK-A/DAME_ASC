from typing import Any, Dict, Iterable


class BaseModel:
    """Abstract interface for models used by the pipeline."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def fit(self, data_iter: Iterable):
        raise NotImplementedError

    def predict(self, sample: Dict) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

