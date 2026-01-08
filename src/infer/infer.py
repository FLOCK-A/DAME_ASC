"""Inference script supporting TTA and per-device ckpt fallback."""
import sys
from pathlib import Path as _Path
# ensure repo root on sys.path
_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import csv
from typing import Dict, Any, Tuple
from pathlib import Path

import numpy as np

from dame_asc.config import load_config
from dame_asc.data.loader import DataLoader
from dame_asc.features import prepare_features
from src.train.train_utils import build_modules, load_checkpoint, softmax


def find_device_ckpt(device_ckpt_dir: str, device_id: int) -> str | None:
    if not device_ckpt_dir:
        return None
    base = Path(device_ckpt_dir)
    if not base.exists():
        return None
    f1 = base / f"device_{device_id}.ckpt"
    if f1.exists():
        return str(f1)
    ddir = base / f"device_{device_id}"
    if ddir.exists():
        f2 = ddir / "best.ckpt"
        if f2.exists():
            return str(f2)
    # also check for directory named with id directly
    ddir2 = base / str(device_id)
    if ddir2.exists():
        f3 = ddir2 / "best.ckpt"
        if f3.exists():
            return str(f3)
    return None


def infer_one_sample(
    sample: Dict[str, Any],
    experts,
    fusion,
    dcdir,
    cfg: Dict[str, Any],
    tta_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    n_crops = int(tta_cfg.get("num_crops", 1)) if tta_cfg.get("enable", False) else 1
    probs_accum = None
    rng = np.random.RandomState(0)
    for _ in range(n_crops):
        features, _, _ = prepare_features([sample], cfg, dcdir, training=False, rng=rng)
        expert_logits = []
        for expert in experts:
            logits, _ = expert.forward(features)
            expert_logits.append(logits[0])
        expert_logits = np.stack(expert_logits, axis=0)[:, None, :]
        if fusion is not None:
            device_raw = sample.get("device", -1)
            device = -1 if device_raw is None else int(device_raw)
            fused_logits, _ = fusion.forward(expert_logits, np.array([device]))
            probs = softmax(fused_logits)[0]
        else:
            probs = softmax(expert_logits.mean(axis=0))[0]
        if probs_accum is None:
            probs_accum = probs
        else:
            probs_accum = probs_accum + probs
    probs_final = probs_accum / float(n_crops)
    pred = int(np.argmax(probs_final))
    return {
        "id": sample.get("id"),
        "path": sample.get("path"),
        "device": -1 if sample.get("device", -1) is None else int(sample.get("device", -1)),
        "pred": pred,
        "probs": probs_final.tolist(),
    }


def load_model(cfg: Dict[str, Any], ckpt_path: str):
    experts, fusion, dcdir = build_modules(cfg)
    load_checkpoint(ckpt_path, experts, fusion, dcdir)
    return experts, fusion, dcdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--general_ckpt", required=True)
    parser.add_argument("--device_ckpt_dir", required=False, default=None)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tta_cfg = cfg.get("infer", {}).get("tta", {}) or {}

    # Load manifest
    loader = DataLoader(args.test_manifest)
    samples = loader.load_manifest()

    # Load general model once
    general_experts, general_fusion, general_dcdir = load_model(cfg, args.general_ckpt)
    device_models: Dict[int, Tuple[Any, Any, Any]] = {}

    # Prepare CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = int(cfg.get("model", {}).get("num_classes", 3))
    header = ["id", "path", "device", "used_ckpt", "pred"] + [f"p_{i}" for i in range(num_classes)]

    with open(str(out_path), "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)
        for s in samples:
            device = -1 if s.get("device", -1) is None else int(s.get("device", -1))
            device_ckpt = find_device_ckpt(args.device_ckpt_dir, device) if device >= 0 else None
            used_ckpt = device_ckpt if device_ckpt else args.general_ckpt
            if device_ckpt:
                if device not in device_models:
                    device_models[device] = load_model(cfg, device_ckpt)
                experts, fusion, dcdir = device_models[device]
            else:
                experts, fusion, dcdir = general_experts, general_fusion, general_dcdir
            res = infer_one_sample(s, experts, fusion, dcdir, cfg, tta_cfg)
            row = [res["id"], res["path"], res["device"], used_ckpt, res["pred"]] + res["probs"]
            writer.writerow(row)

    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
