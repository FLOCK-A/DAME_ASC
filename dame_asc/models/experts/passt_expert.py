from typing import Dict, Any
import numpy as np
from dame_asc.model.base import BaseModel


class PasstExpert(BaseModel):
    """Placeholder Passt expert that returns logits over classes.

    Deterministic per-sample logits generated from sample id and seed.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.num_classes = int(self.config.get("num_classes", 3))
        self.name = "passt"

    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # derive deterministic logits from sample identifier
        sid = sample.get("id")
        if sid is None:
            sid = sample.get("path") or "0"
        seed = abs(hash(str(sid) + self.name)) % (2 ** 32)
        rng = np.random.RandomState(seed)
        logits = rng.randn(self.num_classes).astype(float)
        return {"id": sid, "logits": logits.tolist(), "expert": self.name}
