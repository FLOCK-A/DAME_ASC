import numpy as np

from dame_asc.models.experts.common import NumpyMLPExpert


def test_backward_returns_input_grad_shape():
    expert = NumpyMLPExpert("unit", {"input_dim": 8, "num_classes": 4, "hidden_dims": [6]})
    x = np.random.RandomState(0).randn(3, 8).astype(np.float32)
    logits, cache = expert.forward(x)
    dlogits = np.ones_like(logits)
    grad = expert.backward(dlogits, cache)
    assert grad.shape == x.shape
