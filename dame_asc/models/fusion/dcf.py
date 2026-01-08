from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class DCFusion:
    """Device-Conditioned Fusion with trainable gating and temperature."""

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        self.embed_dim = int(config.get("embed_dim", 32))
        self.hidden = int(config.get("hidden", 64))
        self.use_temperature = bool(config.get("use_temperature", False))
        self.expert_names = config.get("expert_names", ["passt", "cnn"])
        self.num_devices = int(config.get("num_devices", 16))
        self.unknown_index = int(config.get("unknown_index", max(0, self.num_devices - 1)))
        seed = int(config.get("seed", 0))
        rng = np.random.RandomState(seed)
        self.device_embeddings = rng.randn(self.num_devices, self.embed_dim).astype(np.float32) / np.sqrt(self.embed_dim)
        self.g_w1 = rng.randn(self.embed_dim, self.hidden).astype(np.float32) / np.sqrt(self.embed_dim)
        self.g_b1 = np.zeros(self.hidden, dtype=np.float32)
        self.g_w2 = rng.randn(self.hidden, len(self.expert_names)).astype(np.float32) / np.sqrt(self.hidden)
        self.g_b2 = np.zeros(len(self.expert_names), dtype=np.float32)
        if self.use_temperature:
            self.t_w = rng.randn(self.hidden, len(self.expert_names)).astype(np.float32) / np.sqrt(self.hidden)
            self.t_b = np.zeros(len(self.expert_names), dtype=np.float32)
        self.grad_device_embeddings = np.zeros_like(self.device_embeddings)
        self.grad_g_w1 = np.zeros_like(self.g_w1)
        self.grad_g_b1 = np.zeros_like(self.g_b1)
        self.grad_g_w2 = np.zeros_like(self.g_w2)
        self.grad_g_b2 = np.zeros_like(self.g_b2)
        self.grad_t_w = np.zeros_like(self.t_w) if self.use_temperature else None
        self.grad_t_b = np.zeros_like(self.t_b) if self.use_temperature else None

    def _device_index(self, device_id: int) -> int:
        if device_id is None or int(device_id) < 0:
            return self.unknown_index
        if int(device_id) >= self.num_devices:
            return self.unknown_index
        return int(device_id)

    def _forward_single(self, expert_logits: np.ndarray, device_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        idx = self._device_index(device_id)
        embed = self.device_embeddings[idx]
        h = np.tanh(embed @ self.g_w1 + self.g_b1)
        gate_logits = h @ self.g_w2 + self.g_b2
        pi = _softmax(gate_logits)
        if self.use_temperature:
            t_logits = h @ self.t_w + self.t_b
            T = np.log1p(np.exp(t_logits)) + 1.0
        else:
            T = np.ones(len(self.expert_names), dtype=np.float32)
        scaled = expert_logits / T[:, None]
        probs = _softmax_rows(scaled)
        fused_probs = np.sum(pi[:, None] * probs, axis=0)
        fused_logits = np.log(np.clip(fused_probs, 1e-12, 1.0))
        cache = {
            "device_index": idx,
            "embed": embed,
            "h": h,
            "gate_logits": gate_logits,
            "pi": pi,
            "T": T,
            "expert_logits": expert_logits,
            "expert_probs": probs,
            "fused_probs": fused_probs,
        }
        return fused_logits, cache

    def forward(self, expert_logits: np.ndarray, device_ids: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        fused = []
        caches = []
        for i in range(expert_logits.shape[1]):
            f, cache = self._forward_single(expert_logits[:, i, :], int(device_ids[i]))
            fused.append(f)
            caches.append(cache)
        return np.vstack(fused), caches

    def backward(self, dlogits: np.ndarray, caches: List[Dict[str, Any]]) -> np.ndarray:
        num_experts = len(self.expert_names)
        grad_expert_logits = np.zeros((num_experts, dlogits.shape[0], dlogits.shape[1]), dtype=np.float32)
        for i, cache in enumerate(caches):
            pi = cache["pi"]
            T = cache["T"]
            expert_logits = cache["expert_logits"]
            expert_probs = cache["expert_probs"]
            fused_probs = cache["fused_probs"]
            dlog = dlogits[i]
            d_fused = dlog / np.clip(fused_probs, 1e-12, 1.0)
            d_pi = np.array([np.sum(d_fused * expert_probs[k]) for k in range(num_experts)])
            d_z_list = []
            for k in range(num_experts):
                d_probs = pi[k] * d_fused
                d_z = expert_probs[k] * (d_probs - np.sum(d_probs * expert_probs[k]))
                d_z_list.append(d_z)
                grad_expert_logits[k, i, :] = d_z / T[k]
            if self.use_temperature:
                d_T = np.array(
                    [
                        -np.sum(d_z_list[k] * (expert_logits[k] / (T[k] ** 2)))
                        for k in range(num_experts)
                    ]
                )
                t_logits = cache["h"] @ self.t_w + self.t_b
                d_t_logits = d_T * (1.0 / (1.0 + np.exp(-t_logits)))
                self.grad_t_w += np.outer(cache["h"], d_t_logits)
                self.grad_t_b += d_t_logits
                d_h_temp = self.t_w @ d_t_logits
            else:
                d_h_temp = 0.0
            pi_vec = pi
            d_gate_logits = pi_vec * (d_pi - np.sum(d_pi * pi_vec))
            self.grad_g_w2 += np.outer(cache["h"], d_gate_logits)
            self.grad_g_b2 += d_gate_logits
            d_h = self.g_w2 @ d_gate_logits + d_h_temp
            d_h = d_h * (1.0 - cache["h"] ** 2)
            self.grad_g_w1 += np.outer(cache["embed"], d_h)
            self.grad_g_b1 += d_h
            self.grad_device_embeddings[cache["device_index"]] += self.g_w1 @ d_h
        return grad_expert_logits

    def zero_grad(self):
        self.grad_device_embeddings.fill(0.0)
        self.grad_g_w1.fill(0.0)
        self.grad_g_b1.fill(0.0)
        self.grad_g_w2.fill(0.0)
        self.grad_g_b2.fill(0.0)
        if self.use_temperature:
            self.grad_t_w.fill(0.0)
            self.grad_t_b.fill(0.0)

    def parameters(self) -> List[np.ndarray]:
        params = [self.device_embeddings, self.g_w1, self.g_b1, self.g_w2, self.g_b2]
        if self.use_temperature:
            params.extend([self.t_w, self.t_b])
        return params

    def gradients(self) -> List[np.ndarray]:
        grads = [self.grad_device_embeddings, self.grad_g_w1, self.grad_g_b1, self.grad_g_w2, self.grad_g_b2]
        if self.use_temperature:
            grads.extend([self.grad_t_w, self.grad_t_b])
        return grads

    def add_gate_entropy_grad(self, caches: List[Dict[str, Any]], weight: float):
        if weight == 0.0:
            return
        for cache in caches:
            pi = cache["pi"]
            d_pi = -(np.log(np.clip(pi, 1e-12, 1.0)) + 1.0)
            d_gate_logits = pi * (d_pi - np.sum(d_pi * pi))
            self.grad_g_w2 += weight * np.outer(cache["h"], d_gate_logits)
            self.grad_g_b2 += weight * d_gate_logits
            d_h = self.g_w2 @ d_gate_logits
            d_h = d_h * (1.0 - cache["h"] ** 2)
            self.grad_g_w1 += weight * np.outer(cache["embed"], d_h)
            self.grad_g_b1 += weight * d_h
            self.grad_device_embeddings[cache["device_index"]] += weight * (self.g_w1 @ d_h)

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = {
            "device_embeddings": self.device_embeddings,
            "g_w1": self.g_w1,
            "g_b1": self.g_b1,
            "g_w2": self.g_w2,
            "g_b2": self.g_b2,
            "num_devices": np.array([self.num_devices], dtype=np.int32),
            "embed_dim": np.array([self.embed_dim], dtype=np.int32),
            "hidden": np.array([self.hidden], dtype=np.int32),
            "use_temperature": np.array([int(self.use_temperature)], dtype=np.int32),
        }
        if self.use_temperature:
            state["t_w"] = self.t_w
            state["t_b"] = self.t_b
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        self.device_embeddings[...] = state["device_embeddings"]
        self.g_w1[...] = state["g_w1"]
        self.g_b1[...] = state["g_b1"]
        self.g_w2[...] = state["g_w2"]
        self.g_b2[...] = state["g_b2"]
        if self.use_temperature:
            self.t_w[...] = state["t_w"]
            self.t_b[...] = state["t_b"]

    def fuse(self, expert_outputs: List[Dict[str, Any]], device_id: int) -> Dict[str, Any]:
        expert_map = {e["expert"]: np.array(e["logits"], dtype=float) for e in expert_outputs}
        mats = []
        for name in self.expert_names:
            mats.append(expert_map.get(name, np.zeros_like(next(iter(expert_map.values())))))
        mats = np.vstack(mats)
        fused_logits, cache = self._forward_single(mats, int(device_id))
        probs = cache["fused_probs"]
        return {"device_id": int(device_id), "probs": probs.tolist()}

