import numpy as np
from dame_asc.augment.dcdir_bank import MelEQBank


def test_mel_eq_bank_basic():
    n_mels = 64
    bank = MelEQBank(bank_size=4, n_mels=n_mels, max_db=6.0, smooth_kernel=5)
    mel = np.zeros((10, n_mels), dtype=float)
    out = bank.apply_to_mel(mel, device_id=1)
    assert out.shape == mel.shape
    # output should not be all zeros
    assert not np.allclose(out, 0.0)


def test_style_clamp():
    n_mels = 32
    bank = MelEQBank(bank_size=3, n_mels=n_mels, max_db=1.0, smooth_kernel=3)
    style = bank.style_for_device(2)
    assert style.shape[0] == n_mels
    assert style.max() <= 1.0 + 1e-6
    assert style.min() >= -1.0 - 1e-6

