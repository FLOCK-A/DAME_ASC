from typing import Dict, Any, List
import random

from .base import BaseModel


class PlaceholderModel(BaseModel):
    """A deterministic/random placeholder model for smoke tests."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.labels: List[str] = self.config.get("labels", ["scene_a", "scene_b", "scene_c"])  # default labels
        seed = self.config.get("seed")
        if seed is not None:
            random.seed(seed)

    def fit(self, data_iter):
        # no-op for placeholder
        return None

    def predict(self, sample: Dict) -> Dict[str, Any]:
        # Return a pseudo-random label with a confidence
        label = random.choice(self.labels)
        score = random.random()
        return {"id": sample.get("id"), "pred": label, "score": score}

    def save(self, path: str):
        # no-op
        return None

    def load(self, path: str):
        # no-op
        return None

