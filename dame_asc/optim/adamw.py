from __future__ import annotations

from typing import List, Tuple

import numpy as np


class AdamW:
    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.state = []
        self._init_state()
        self.step_count = 0

    def _init_state(self):
        self.state = []
        for p in self.params:
            self.state.append({"m": np.zeros_like(p), "v": np.zeros_like(p)})

    def set_params(self, params: List[np.ndarray]):
        if len(params) != len(self.params):
            self.params = params
            self._init_state()
            self.step_count = 0
        else:
            self.params = params

    def step(self, grads: List[np.ndarray]):
        self.step_count += 1
        lr = self.lr
        for idx, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * p
            m = self.state[idx]["m"]
            v = self.state[idx]["v"]
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            m_hat = m / (1.0 - self.beta1 ** self.step_count)
            v_hat = v / (1.0 - self.beta2 ** self.step_count)
            p[:] = p - lr * m_hat / (np.sqrt(v_hat) + self.eps)
