from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


@dataclass
class ExpertCache:
    inputs: np.ndarray
    preacts: List[np.ndarray]
    activations: List[np.ndarray]


class NumpyMLPExpert:
    """Lightweight numpy MLP expert with trainable parameters."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config or {}
        self.input_dim = int(self.config.get("input_dim", 128))
        self.num_classes = int(self.config.get("num_classes", 3))
        self.hidden_dims = [int(h) for h in self.config.get("hidden_dims", [256, 128])]
        seed = int(self.config.get("seed", 0)) + abs(hash(name)) % 1000
        rng = np.random.RandomState(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.grads_w: List[np.ndarray] = []
        self.grads_b: List[np.ndarray] = []
        dims = [self.input_dim] + self.hidden_dims + [self.num_classes]
        for din, dout in zip(dims[:-1], dims[1:]):
            w = rng.randn(din, dout).astype(np.float32) / np.sqrt(din)
            b = np.zeros(dout, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
            self.grads_w.append(np.zeros_like(w))
            self.grads_b.append(np.zeros_like(b))

    def forward(self, features: np.ndarray) -> Tuple[np.ndarray, ExpertCache]:
        x = features
        preacts = []
        activations = [x]
        for idx in range(len(self.weights)):
            z = x @ self.weights[idx] + self.biases[idx]
            preacts.append(z)
            if idx < len(self.weights) - 1:
                x = relu(z)
                activations.append(x)
            else:
                x = z
        cache = ExpertCache(inputs=features, preacts=preacts, activations=activations)
        return x, cache

    def backward(self, dlogits: np.ndarray, cache: ExpertCache) -> np.ndarray:
        grad = dlogits
        for idx in reversed(range(len(self.weights))):
            a_prev = cache.activations[idx]
            self.grads_w[idx] = a_prev.T @ grad / float(a_prev.shape[0])
            self.grads_b[idx] = grad.mean(axis=0)
            grad = grad @ self.weights[idx].T
            if idx > 0:
                grad = grad * relu_grad(cache.preacts[idx - 1])
        return grad

    def zero_grad(self):
        for idx in range(len(self.grads_w)):
            self.grads_w[idx].fill(0.0)
            self.grads_b[idx].fill(0.0)

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases

    def gradients(self) -> List[np.ndarray]:
        return self.grads_w + self.grads_b

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = {}
        for idx, w in enumerate(self.weights):
            state[f"w{idx}"] = w
            state[f"b{idx}"] = self.biases[idx]
        state["input_dim"] = np.array([self.input_dim], dtype=np.int32)
        state["num_classes"] = np.array([self.num_classes], dtype=np.int32)
        state["hidden_dims"] = np.array(self.hidden_dims, dtype=np.int32)
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        for idx in range(len(self.weights)):
            self.weights[idx][...] = state[f"w{idx}"]
            self.biases[idx][...] = state[f"b{idx}"]
