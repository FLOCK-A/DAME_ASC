"""Inference script supporting TTA and per-device ckpt fallback.

Behavior:
- Load config (infer settings), load test manifest.
- For each sample: if device>=0 and a device-specific ckpt exists -> use stage-2 (device) model; else use general stage-1 model.
- Build experts from config.model.experts and build fusion from config.model.fusion.
- Apply TTA by repeating predictions num_crops times and averaging probabilities.
- Output CSV with id,path,device,pred,probs...,used_ckpt

Notes/Assumptions:
- Device ckpt lookup: if DEVICE_CKPT_DIR contains file named "device_{id}.ckpt" or folder named "device_{id}" with a file "best.ckpt", we treat that as a stage-2 ckpt. If not found, use general_ckpt.
- For placeholder models, ckpt files are not actually loaded; we only use their presence to choose stage-2 vs stage-1 path.
"""
import sys
from pathlib import Path as _Path
# ensure repo root on sys.path
_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import csv
from typing import Dict, Any
from pathlib import Path

import numpy as np

from dame_asc.config import load_config
from dame_asc.data.loader import DataLoader
from dame_asc.models.factory import build_expert, build_fusion


def softmax(logits: np.ndarray) -> np.ndarray:
    l = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(axis=-1, keepdims=True)


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


def infer_one_sample(sample: Dict[str, Any], experts, fusion, tta_cfg: Dict[str, Any]) -> Dict[str, Any]:
    n_crops = int(tta_cfg.get("num_crops", 1)) if tta_cfg.get("enable", False) else 1
    probs_accum = None
    for c in range(n_crops):
        expert_outputs = [ex.predict(sample) for ex in experts]
        if fusion is not None:
            fused = fusion.fuse(expert_outputs, int(sample.get("device", -1) or -1))
            probs = np.array(fused["probs"])
        else:
            # average softmax of logits
            mats = np.vstack([np.array(o["logits"]) for o in expert_outputs])
            probs = softmax(mats.mean(axis=0))
        if probs_accum is None:
            probs_accum = probs
        else:
            probs_accum = probs_accum + probs
    probs_final = probs_accum / float(n_crops)
    pred = int(np.argmax(probs_final))
    return {"id": sample.get("id"), "path": sample.get("path"), "device": int(sample.get("device", -1) or -1), "pred": pred, "probs": probs_final.tolist()}


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

    # Build base experts and fusion from config
    expert_cfgs = cfg.get("model", {}).get("experts", [{"name": "passt"}, {"name": "cnn"}])
    fusion_cfg = cfg.get("model", {}).get("fusion", None)

    # We'll build experts and fusion once and reuse; for device-specific ckpt presence we only switch which_ckpt reported
    experts = [build_expert(e.get("name") if isinstance(e, dict) else e, e if isinstance(e, dict) else {}) for e in expert_cfgs]
    fusion = build_fusion(fusion_cfg.get("name"), fusion_cfg) if fusion_cfg else None

    # Prepare CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = int(cfg.get("model", {}).get("num_classes", 3))
    header = ["id", "path", "device", "used_ckpt", "pred"] + [f"p_{i}" for i in range(num_classes)]

    with open(str(out_path), "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)
        for s in samples:
            device = int(s.get("device", -1) or -1)
            device_ckpt = find_device_ckpt(args.device_ckpt_dir, device) if device >= 0 else None
            used_ckpt = device_ckpt if device_ckpt else args.general_ckpt

            # (Placeholder) In a real system, you'd load device-specific model weights here. For now, use same experts/fusion but report used_ckpt.
            res = infer_one_sample(s, experts, fusion, tta_cfg)
            row = [res["id"], res["path"], res["device"], used_ckpt, res["pred"]] + res["probs"]
            writer.writerow(row)

    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
