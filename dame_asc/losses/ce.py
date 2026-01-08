import numpy as np
from typing import Tuple


def softmax(logits: np.ndarray) -> np.ndarray:
    l = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_with_label_smoothing(logits: np.ndarray, target: int, num_classes: int, label_smoothing: float = 0.0) -> float:
    """Compute cross-entropy for a single sample with optional label smoothing.

    logits: (C,) or (1,C)
    target: int
    """
    probs = softmax(np.asarray(logits))
    assert probs.shape[-1] == num_classes
    if label_smoothing and label_smoothing > 0.0:
        eps = float(label_smoothing)
        smoothed = (1.0 - eps) * np.eye(num_classes)[int(target)] + eps / float(num_classes)
        loss = -np.sum(smoothed * np.log(np.clip(probs, 1e-12, 1.0)))
    else:
        loss = -np.log(np.clip(probs[int(target)], 1e-12, 1.0))
    return float(loss)


def batch_ce(logits_batch: np.ndarray, targets: np.ndarray, label_smoothing: float = 0.0) -> float:
    """Compute mean CE over batch. logits_batch: (N,C), targets: (N,)
    """
    N, C = logits_batch.shape
    losses = []
    for i in range(N):
        losses.append(cross_entropy_with_label_smoothing(logits_batch[i], int(targets[i]), C, label_smoothing))
    return float(np.mean(losses))
