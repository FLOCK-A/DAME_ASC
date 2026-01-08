from typing import Dict, Sequence, Tuple

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


class MelEQBank:
    """Trainable device-conditioned mel EQ bank."""

    def __init__(
        self,
        bank_size: int = 8,
        n_mels: int = 128,
        max_db: float = 6.0,
        smooth_kernel: int = 9,
        embed_dim: int = 16,
        num_devices: int = 16,
        unknown_index: int | None = None,
        seed: int = 0,
    ):
        self.bank_size = int(bank_size)
        self.n_mels = int(n_mels)
        self.max_db = float(max_db)
        self.smooth_kernel = int(smooth_kernel)
        self.embed_dim = int(embed_dim)
        self.num_devices = int(num_devices)
        if unknown_index is None:
            self.unknown_index = max(0, self.num_devices - 1)
        else:
            self.unknown_index = int(unknown_index)
        rng = np.random.RandomState(seed)
        base = rng.randn(self.bank_size, self.n_mels) * (self.max_db / 2.0)
        self.prototypes = np.array([self._smooth_row(r) for r in base]).astype(np.float32)
        self.device_embeddings = rng.randn(self.num_devices, self.embed_dim).astype(np.float32) / np.sqrt(self.embed_dim)
        self.proj_w = rng.randn(self.embed_dim, self.bank_size).astype(np.float32) / np.sqrt(self.embed_dim)
        self.proj_b = np.zeros(self.bank_size, dtype=np.float32)
        self.grad_prototypes = np.zeros_like(self.prototypes)
        self.grad_device_embeddings = np.zeros_like(self.device_embeddings)
        self.grad_proj_w = np.zeros_like(self.proj_w)
        self.grad_proj_b = np.zeros_like(self.proj_b)

    def _smooth_row(self, row: Sequence[float]) -> np.ndarray:
        k = max(1, self.smooth_kernel)
        pad = k // 2
        arr = np.pad(np.asarray(row), (pad, pad), mode="reflect")
        kernel = np.ones(k) / k
        sm = np.convolve(arr, kernel, mode="valid")
        return sm[: self.n_mels]

    def _device_index(self, device_id: int) -> int:
        if device_id is None or int(device_id) < 0:
            return self.unknown_index
        if int(device_id) >= self.num_devices:
            return self.unknown_index
        return int(device_id)

    def style_for_device(self, device_id: int) -> np.ndarray:
        idx = self._device_index(device_id)
        emb = self.device_embeddings[idx]
        logits = emb @ self.proj_w + self.proj_b
        w = _softmax(logits)
        raw_style = w @ self.prototypes
        style = self.max_db * np.tanh(raw_style / self.max_db)
        return style

    def apply_to_mel(self, mel: np.ndarray, device_id: int, return_cache: bool = False) -> Tuple[np.ndarray, Dict] | np.ndarray:
        if mel.ndim not in (1, 2):
            raise ValueError("mel must be 1D or 2D array")
        if mel.shape[-1] != self.n_mels:
            raise ValueError("n_mels mismatch")
        idx = self._device_index(device_id)
        emb = self.device_embeddings[idx]
        logits = emb @ self.proj_w + self.proj_b
        w = _softmax(logits)
        raw_style = w @ self.prototypes
        style = self.max_db * np.tanh(raw_style / self.max_db)
        out = mel + style if mel.ndim == 1 else mel + style[np.newaxis, :]
        if not return_cache:
            return out
        cache = {
            "device_index": idx,
            "weights": w,
            "raw_style": raw_style,
            "style": style,
        }
        return out, cache

    def backward(self, grad_aug: np.ndarray, cache: Dict[str, np.ndarray]):
        if grad_aug.ndim == 2:
            grad_style = grad_aug.sum(axis=0)
        else:
            grad_style = grad_aug
        raw_style = cache["raw_style"]
        weights = cache["weights"]
        device_index = cache["device_index"]
        d_raw = grad_style * (1.0 - np.tanh(raw_style / self.max_db) ** 2)
        self.grad_prototypes += np.outer(weights, d_raw)
        d_weights = self.prototypes @ d_raw
        w = weights
        d_logits = w * (d_weights - np.sum(d_weights * w))
        self.grad_proj_w += np.outer(self.device_embeddings[device_index], d_logits)
        self.grad_proj_b += d_logits
        self.grad_device_embeddings[device_index] += self.proj_w @ d_logits

    def zero_grad(self):
        self.grad_prototypes.fill(0.0)
        self.grad_device_embeddings.fill(0.0)
        self.grad_proj_w.fill(0.0)
        self.grad_proj_b.fill(0.0)

    def parameters(self):
        return [self.prototypes, self.device_embeddings, self.proj_w, self.proj_b]

    def gradients(self):
        return [self.grad_prototypes, self.grad_device_embeddings, self.grad_proj_w, self.grad_proj_b]

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "prototypes": self.prototypes,
            "device_embeddings": self.device_embeddings,
            "proj_w": self.proj_w,
            "proj_b": self.proj_b,
            "bank_size": np.array([self.bank_size], dtype=np.int32),
            "n_mels": np.array([self.n_mels], dtype=np.int32),
            "max_db": np.array([self.max_db], dtype=np.float32),
            "embed_dim": np.array([self.embed_dim], dtype=np.int32),
            "num_devices": np.array([self.num_devices], dtype=np.int32),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        self.prototypes[...] = state["prototypes"]
        self.device_embeddings[...] = state["device_embeddings"]
        self.proj_w[...] = state["proj_w"]
        self.proj_b[...] = state["proj_b"]

