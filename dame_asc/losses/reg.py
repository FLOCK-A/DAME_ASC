import numpy as np
from typing import Iterable


def gate_entropy(pi: np.ndarray) -> float:
    """Entropy of gating distribution pi (K,)."""
    p = np.asarray(pi, dtype=float)
    p = p / p.sum()
    return float(- (p * np.log(np.clip(p, 1e-12, 1.0))).sum())


def dcdir_l2_norm(prototypes: Iterable[np.ndarray]) -> float:
    """Compute L2 norm regularization over DCDIR prototypes (list of arrays)."""
    s = 0.0
    count = 0
    for p in prototypes:
        a = np.asarray(p, dtype=float)
        s += float((a ** 2).sum())
        count += a.size
    if count == 0:
        return 0.0
    return float(s / count)
