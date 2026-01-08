from typing import Dict, Any, List
import numpy as np


class DCFusion:
    """Device-Conditioned Fusion (placeholder simple implementation).

    This placeholder uses a device embedding (device_id -> vector), a small MLP
    to compute gating logits over experts and an optional temperature per expert.
    It accepts expert outputs which are dicts: {"id":..., "logits": [...], "expert": name}
    and returns fused probabilities over classes.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        self.embed_dim = int(config.get("embed_dim", 32))
        self.hidden = int(config.get("hidden", 64))
        self.use_temperature = bool(config.get("use_temperature", False))
        self.expert_names = config.get("expert_names", ["passt", "cnn"])  # default
        # random but deterministic parameters
        rng = np.random.RandomState(0)
        self.device_embeddings = rng.randn(256, self.embed_dim)  # support device ids up to 255
        self.g_w1 = rng.randn(self.embed_dim, self.hidden)
        self.g_b1 = rng.randn(self.hidden)
        self.g_w2 = rng.randn(self.hidden, len(self.expert_names))
        self.g_b2 = rng.randn(len(self.expert_names))
        if self.use_temperature:
            self.t_w = rng.randn(self.hidden, len(self.expert_names))
            self.t_b = rng.randn(len(self.expert_names))

    def _device_embed(self, device_id: int) -> np.ndarray:
        di = int(device_id) % len(self.device_embeddings)
        return self.device_embeddings[di]

    def _mlp_gating(self, embed: np.ndarray) -> np.ndarray:
        h = embed.dot(self.g_w1) + self.g_b1
        h = np.tanh(h)
        logits = h.dot(self.g_w2) + self.g_b2
        exps = np.exp(logits - logits.max())
        return exps / exps.sum()

    def _mlp_temperature(self, embed: np.ndarray) -> np.ndarray:
        h = embed.dot(self.g_w1) + self.g_b1
        h = np.tanh(h)
        t = h.dot(self.t_w) + self.t_b
        # softplus + 1
        return np.log1p(np.exp(t)) + 1.0

    def fuse(self, expert_outputs: List[Dict[str, Any]], device_id: int) -> Dict[str, Any]:
        # reorganize logits per expert and ensure same class dim
        expert_map = {e["expert"]: np.array(e["logits"], dtype=float) for e in expert_outputs}
        K = len(self.expert_names)
        # build matrix K x C
        mats = []
        for name in self.expert_names:
            mats.append(expert_map.get(name, np.zeros_like(next(iter(expert_map.values())))))
        mats = np.vstack(mats)  # K x C

        embed = self._device_embed(device_id)
        pi = self._mlp_gating(embed)  # K
        if self.use_temperature:
            T = self._mlp_temperature(embed)  # K
        else:
            T = np.ones(K)

        # apply temperature and softmax per expert
        probs = []
        for k in range(K):
            logits = mats[k] / T[k]
            ex = np.exp(logits - logits.max())
            probs.append(ex / ex.sum())
        probs = np.vstack(probs)  # K x C

        # weighted sum
        final = (pi[:, None] * probs).sum(axis=0)
        return {"device_id": int(device_id), "pi": pi.tolist(), "probs": final.tolist()}

