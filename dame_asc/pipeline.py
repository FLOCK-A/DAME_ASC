from typing import Dict, Any

from .data.loader import DataLoader
from .utils import write_json
from .model.base import BaseModel


def run_inference(cfg: Dict[str, Any], model: BaseModel) -> Dict[str, Any]:
    # Prepare data
    inference_cfg = cfg.get("inference", {})
    synthetic_count = int(inference_cfg.get("synthetic_count", 0))

    dataset_cfg = cfg.get("dataset", {}) or {}
    manifest_path = dataset_cfg.get("manifest")
    manifest_format = dataset_cfg.get("manifest_format", "dict")

    loader = DataLoader(manifest_path, manifest_format)

    if synthetic_count > 0:
        samples = loader.generate_synthetic(synthetic_count)
    else:
        samples = loader.list_samples()

    # Optional augmentation: device-conditioned DIR bank (mel-eq)
    augment_cfg = cfg.get("augment", {}) or {}
    dcdir_cfg = augment_cfg.get("dcdir", {}) if isinstance(augment_cfg, dict) else {}
    dcdir_enable = bool(dcdir_cfg.get("enable", False))

    use_mel_eq = dcdir_enable and dcdir_cfg.get("mode") == "mel_eq_bank"

    if use_mel_eq:
        try:
            from .augment.dcdir_bank import MelEQBank
            import numpy as np
        except Exception:
            # If numpy or module unavailable, disable augmentation quietly
            use_mel_eq = False

    # Determine n_mels from input config if present
    input_cfg = cfg.get("input", {}) or {}
    n_mels = int(input_cfg.get("n_mels", 128))
    n_frames = int(input_cfg.get("n_frames", 10))

    if use_mel_eq:
        bank = MelEQBank(bank_size=int(dcdir_cfg.get("bank_size", 8)), n_mels=n_mels,
                         max_db=float(dcdir_cfg.get("max_db", 6.0)),
                         smooth_kernel=int(dcdir_cfg.get("smooth_kernel", 9)))

    predictions = []
    for s in samples:
        # If augmentation requested and we have no real features, build synthetic mel and apply
        if use_mel_eq:
            # device id fallback
            device_id = int(s.get("device", -1) or -1)
            # create zero mel and apply
            mel = np.zeros((n_frames, n_mels), dtype=float)
            aug_mel = bank.apply_to_mel(mel, device_id)
            # store as metadata (model may use this later)
            s_meta = s.get("meta") or {}
            s_meta["aug_mel_shape"] = aug_mel.shape
            s["meta"] = s_meta

        pred = model.predict(s)
        predictions.append(pred)

    out = {"n_samples": len(samples), "predictions": predictions}

    out_path = inference_cfg.get("output", "inference_results.json")
    write_json(out_path, out)
    return out
