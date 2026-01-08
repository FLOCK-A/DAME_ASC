from typing import Dict, Any

import numpy as np

from dame_asc.model.base import BaseModel
from .common import NumpyMLPExpert


class BeatsExpert(BaseModel):
    """Lightweight BEATs-style expert with a deeper MLP stack."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        cfg = self.config.copy()
        cfg.setdefault("hidden_dims", [768, 384, 192])
        self.model = NumpyMLPExpert("beats", cfg)
        self.name = "beats"

    def forward(self, features: np.ndarray):
        return self.model.forward(features)

    def backward(self, dlogits: np.ndarray, cache):
        return self.model.backward(dlogits, cache)

    def zero_grad(self):
        self.model.zero_grad()

    def parameters(self):
        return self.model.parameters()

    def gradients(self):
        return self.model.gradients()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        features = sample.get("features")
        if features is None:
            raise ValueError("features missing in sample for BeatsExpert.predict")
        logits, _ = self.forward(np.asarray(features)[None, :])
        return {"id": sample.get("id"), "logits": logits[0].tolist(), "expert": self.name}
