"""Minimal placeholder training script for Stage-2 (device-specific fine-tuning).

This implements Option A: freeze experts and fine-tune fusion (placeholder).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from dame_asc.config import load_config
from dame_asc.data.loader import DataLoader
from dame_asc.models.factory import build_expert, build_fusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_manifest", required=False)
    p.add_argument("--val_manifest", required=False)
    p.add_argument("--init_ckpt", required=False)
    p.add_argument("--device_id", type=int, required=False)
    p.add_argument("--all_devices", action="store_true")
    p.add_argument("--workdir", required=False, default="runs/stage2")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    workdir = args.workdir

    manifest = args.train_manifest or cfg.get("dataset", {}).get("manifest")
    loader = DataLoader(manifest)
    samples = loader.generate_synthetic(4) if (not manifest) else loader.load_manifest()

    # Build experts (and freeze them conceptually)
    expert_cfgs = cfg.get("model", {}).get("experts", [{"name": "passt"}, {"name": "cnn"}])
    experts = []
    for e in expert_cfgs:
        name = e.get("name") if isinstance(e, dict) else e
        experts.append(build_expert(name, e if isinstance(e, dict) else {}))

    # Build fusion
    fusion_cfg = cfg.get("model", {}).get("fusion", {"name": "dcf"})
    fusion = build_fusion(fusion_cfg.get("name", "dcf"), fusion_cfg)

    # Placeholder fine-tune loop: just evaluate fusion on existing experts
    results = []
    for s in samples:
        expert_outputs = [ex.predict(s) for ex in experts]
        device = int(s.get("device", -1) or -1)
        fused = fusion.fuse(expert_outputs, device)
        # compute ce against ground truth using fused probs
        probs = np.array(fused["probs"])
        pred = int(np.argmax(probs))
        results.append({"id": s.get("id"), "device": device, "pred": pred, "probs": fused["probs"]})

    Path(workdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(workdir, "stage2_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Stage2 results written to {workdir}/stage2_results.json")


if __name__ == "__main__":
    main()

