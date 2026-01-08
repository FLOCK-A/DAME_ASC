import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for discrete distributions p and q (1D arrays).
    Both p and q should sum to 1.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(q, 1e-12, 1.0)))).sum())


def consistency_loss(p_logits: np.ndarray, q_logits: np.ndarray, loss_type: str = "kl") -> float:
    """Compute consistency loss between two logits: KL(p||q) or symmetric MSE on probs.
    p_logits, q_logits: (C,) arrays
    """
    from .ce import softmax

    p = softmax(p_logits)
    q = softmax(q_logits)
    if loss_type == "kl":
        return kl_divergence(p, q)
    elif loss_type == "mse":
        return float(((p - q) ** 2).mean())
    else:
        raise ValueError("Unsupported consistency loss type")

