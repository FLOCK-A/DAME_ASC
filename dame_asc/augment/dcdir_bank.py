from typing import List, Sequence
import numpy as np


class MelEQBank:
    """Device-Conditioned Mel-EQ Bank (Option A).

    Each prototype h_k is a gain curve over n_mels (in dB). For a device d,
    we compute weights w_k(d) (here simulated via a simple softmax over a device id mapping)
    and produce h(d) = sum_k w_k * h_k. The augmentation is mel + clamp(h(d), -max_db, +max_db).

    This is a lightweight, deterministic implementation suitable for unit tests.
    """

    def __init__(self, bank_size: int = 8, n_mels: int = 128, max_db: float = 6.0, smooth_kernel: int = 9):
        self.bank_size = int(bank_size)
        self.n_mels = int(n_mels)
        self.max_db = float(max_db)
        self.smooth_kernel = int(smooth_kernel)
        # prototypes: bank_size x n_mels
        rng = np.random.RandomState(0)
        base = rng.randn(self.bank_size, self.n_mels) * (self.max_db / 2.0)
        # smooth each prototype with a simple moving average
        self.prototypes = np.array([self._smooth_row(r) for r in base])

    def _smooth_row(self, row: Sequence[float]) -> np.ndarray:
        k = max(1, self.smooth_kernel)
        pad = k // 2
        arr = np.pad(np.asarray(row), (pad, pad), mode="reflect")
        kernel = np.ones(k) / k
        sm = np.convolve(arr, kernel, mode="valid")
        return sm[: self.n_mels]

    def _device_weights(self, device_id: int) -> np.ndarray:
        # Simple deterministic mapping: use device_id hashed to bank_size logits
        logits = np.arange(self.bank_size).astype(float) - (device_id % self.bank_size)
        exps = np.exp(logits - logits.max())
        return exps / exps.sum()

    def style_for_device(self, device_id: int) -> np.ndarray:
        w = self._device_weights(device_id)
        # weighted sum: (bank_size,) dot (bank_size, n_mels) -> (n_mels,)
        style = w.dot(self.prototypes)
        # clamp
        style = np.clip(style, -self.max_db, self.max_db)
        return style

    def apply_to_mel(self, mel: np.ndarray, device_id: int) -> np.ndarray:
        """Apply device-conditioned EQ to mel spectrogram.

        mel: (n_frames, n_mels) or (n_mels,) -> returns same shape
        """
        if mel.ndim == 2:
            frames, nm = mel.shape
            assert nm == self.n_mels, "n_mels mismatch"
            style = self.style_for_device(device_id)
            # add style (dB) to each frame
            out = mel + style[np.newaxis, :]
            return out
        elif mel.ndim == 1:
            assert mel.shape[0] == self.n_mels
            style = self.style_for_device(device_id)
            return mel + style
        else:
            raise ValueError("mel must be 1D or 2D array")

