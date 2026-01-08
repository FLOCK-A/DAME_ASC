from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sample_key(sample: Dict[str, Any]) -> str:
    return str(sample.get("id") or sample.get("path") or "0")


def deterministic_mel(sample: Dict[str, Any], n_frames: int, n_mels: int, seed_offset: int = 0) -> np.ndarray:
    seed = (abs(hash(_sample_key(sample))) + seed_offset) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.randn(n_frames, n_mels).astype(np.float32)


def mel_to_feature(mel: np.ndarray) -> np.ndarray:
    return mel.mean(axis=0)


def _load_mel(sample: Dict[str, Any], n_frames: int, n_mels: int, training: bool) -> np.ndarray:
    feat = sample.get("features") or sample.get("feature")
    if feat is not None:
        arr = np.asarray(feat, dtype=np.float32)
        if arr.ndim == 1:
            return arr[None, :]
        return arr
    path = sample.get("path")
    if path:
        p = Path(str(path))
        if p.exists() and p.suffix.lower() == ".npy":
            arr = np.load(str(p))
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                return arr[None, :]
            return arr
    if training:
        raise ValueError(
            "Training samples must include precomputed log-mel features or a .npy path. "
            "Please extract log-mel features offline before training."
        )
    return deterministic_mel(sample, n_frames=n_frames, n_mels=n_mels)


def prepare_features(
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    dcdir_bank: Optional[Any] = None,
    training: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, List[Optional[Dict[str, Any]]], List[bool]]:
    input_cfg = cfg.get("input", {}) or {}
    n_mels = int(input_cfg.get("n_mels", 128))
    n_frames = int(input_cfg.get("n_frames", 10))
    augment_cfg = cfg.get("augment", {}) or {}
    dcdir_cfg = augment_cfg.get("dcdir", {}) if isinstance(augment_cfg, dict) else {}
    dcdir_enable = bool(dcdir_cfg.get("enable", False))
    p_apply = float(dcdir_cfg.get("p", 1.0))
    apply_in_infer = bool(dcdir_cfg.get("apply_in_infer", False))
    tta_cfg = cfg.get("infer", {}).get("tta", {}) or {}
    tta_enable = bool(tta_cfg.get("enable", False))
    tta_time_shift = bool(tta_cfg.get("time_shift", False))

    if rng is None:
        rng = np.random.RandomState(0)

    features = []
    dcdir_caches: List[Optional[Dict[str, Any]]] = []
    applied_flags: List[bool] = []
    for sample in samples:
        mel = _load_mel(sample, n_frames=n_frames, n_mels=n_mels, training=training)
        cache = None
        applied = False
        if dcdir_bank is not None and dcdir_enable:
            if (training and rng.rand() <= p_apply) or (apply_in_infer and rng.rand() <= p_apply):
                device_raw = sample.get("device", -1)
                device_id = -1 if device_raw is None else int(device_raw)
                mel, cache = dcdir_bank.apply_to_mel(mel, device_id, return_cache=True)
                applied = True
        if (not training) and tta_enable and tta_time_shift:
            if mel.ndim == 2 and mel.shape[0] > 1:
                shift = int(rng.randint(0, mel.shape[0]))
                mel = np.roll(mel, shift, axis=0)
        features.append(mel_to_feature(mel))
        dcdir_caches.append(cache)
        applied_flags.append(applied)
    return np.vstack(features), dcdir_caches, applied_flags
