import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .augment.dcdir_bank import MelEQBank


def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def map_device_index(device_id: int | None, num_devices: int, unknown_index: int) -> int:
    if device_id is None:
        return int(unknown_index)
    device_int = int(device_id)
    if device_int < 0 or device_int >= int(num_devices):
        return int(unknown_index)
    return device_int


def load_feature(
    sample: Dict[str, Any],
    input_cfg: Optional[Dict[str, Any]] = None,
    dcdir_bank: Optional["MelEQBank"] = None,
) -> np.ndarray:
    """Load or synthesize feature vector for a sample.

    Supports:
      - sample["feature"] or sample["features"] already populated
      - .npy file on disk
      - fallback to zeros with configured n_mels/n_frames
    """
    input_cfg = input_cfg or {}
    n_mels = int(input_cfg.get("n_mels", 128))
    n_frames = int(input_cfg.get("n_frames", 10))

    feat = sample.get("feature")
    if feat is None:
        feat = sample.get("features")

    if feat is None:
        path = sample.get("path")
        if path:
            p = Path(str(path))
            if p.exists() and p.suffix.lower() == ".npy":
                feat = np.load(str(p))

    if feat is None:
        feat = np.zeros((n_frames, n_mels), dtype=float)

    feat = np.asarray(feat, dtype=float)

    if feat.ndim == 2 and dcdir_bank is not None:
        device_id = sample.get("device", -1)
        device = -1 if device_id is None else int(device_id)
        feat = dcdir_bank.apply_to_mel(feat, device)

    if feat.ndim == 2:
        # average over time frames
        feat = feat.mean(axis=0)

    return feat.astype(float)

