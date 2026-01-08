from typing import Dict, Any, Iterable, List, Optional, Tuple

import numpy as np

from dame_asc.model.base import BaseModel


def _softmax(logits: np.ndarray) -> np.ndarray:
    l = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


class LinearExpert(BaseModel):
    """Simple trainable linear classifier expert.

    This expert consumes a feature vector and outputs logits for num_classes.
    """

    def __init__(self, config: Dict[str, Any] | None = None, name: str = "linear"):
        super().__init__(config)
        self.name = name
        self.num_classes = int(self.config.get("num_classes", 3))
        self.input_dim: Optional[int] = self.config.get("input_dim")
        self._init_params(seed=self.config.get("seed", 0))

    def _init_params(self, seed: int = 0):
        if self.input_dim is None:
            self.weights = None
            self.bias = None
            return
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(self.input_dim, self.num_classes).astype(float) * 0.01
        self.bias = np.zeros(self.num_classes, dtype=float)

    def _ensure_params(self, input_dim: int):
        if self.weights is None or self.bias is None or self.input_dim != input_dim:
            self.input_dim = int(input_dim)
            self._init_params(seed=self.config.get("seed", 0))

    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        feat = sample.get("feature")
        if feat is None:
            raise ValueError("Sample missing feature vector for expert prediction.")
        x = np.asarray(feat, dtype=float)
        self._ensure_params(x.shape[-1])
        logits = x.dot(self.weights) + self.bias
        return {"id": sample.get("id"), "logits": logits.tolist(), "expert": self.name}

    def fit(
        self,
        dataset: Iterable[Tuple[np.ndarray, int]],
        epochs: int = 5,
        lr: float = 1e-2,
        label_smoothing: float = 0.0,
    ):
        data_list = list(dataset)
        if not data_list:
            return
        input_dim = data_list[0][0].shape[-1]
        self._ensure_params(input_dim)

        for _ in range(int(epochs)):
            for x, y in data_list:
                logits = x.dot(self.weights) + self.bias
                probs = _softmax(logits)
                if label_smoothing > 0.0:
                    eps = float(label_smoothing)
                    smooth = (1.0 - eps) * np.eye(self.num_classes)[int(y)] + eps / float(self.num_classes)
                    grad = probs - smooth
                else:
                    grad = probs
                    grad[int(y)] -= 1.0
                self.weights -= lr * np.outer(x, grad)
                self.bias -= lr * grad

    def save(self, path: str):
        if self.weights is None or self.bias is None:
            raise ValueError("Cannot save uninitialized expert parameters.")
        np.savez(path, weights=self.weights, bias=self.bias, num_classes=self.num_classes, name=self.name)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.weights = data["weights"].astype(float)
        self.bias = data["bias"].astype(float)
        self.num_classes = int(data["num_classes"])
        self.input_dim = int(self.weights.shape[0])
