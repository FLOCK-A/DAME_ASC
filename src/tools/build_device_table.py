"""CLI wrapper to produce per-expert per-device accuracy CSV.

Usage:
  python -m src.tools.build_device_table --val_manifest data/manifests/train.json --out_csv runs/per_device_table.csv

This uses dame_asc.models.factory to build experts and compute acc per device.
"""
import sys
from pathlib import Path as _Path
# ensure repo root (two levels up from src/tools)
_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import csv
import json
from typing import Dict, Any
from pathlib import Path

from dame_asc.data.loader import DataLoader
from dame_asc.features import prepare_features
from dame_asc.models.factory import build_expert


def argmax_idx(logits):
    return int(max(range(len(logits)), key=lambda i: logits[i]))


def compute_table(manifest_path: str, expert_cfgs: list):
    loader = DataLoader(manifest_path)
    samples = loader.load_manifest()
    expert_names = [e.get("name") if isinstance(e, dict) else e for e in expert_cfgs]
    table = {ename: {} for ename in expert_names}
    input_dim = 128
    for e in expert_cfgs:
        if isinstance(e, dict) and "input_dim" in e:
            input_dim = int(e["input_dim"])
            break
    cfg = {"input": {"n_mels": input_dim, "n_frames": 10}, "augment": {}}
    experts = []
    for e in expert_cfgs:
        name = e.get("name") if isinstance(e, dict) else e
        e_cfg = dict(e) if isinstance(e, dict) else {}
        e_cfg.setdefault("input_dim", input_dim)
        experts.append(build_expert(name, e_cfg))

    features, _, _ = prepare_features(samples, cfg, dcdir_bank=None, training=False)
    for idx, s in enumerate(samples):
        device_raw = s.get("device", -1)
        device = -1 if device_raw is None else int(device_raw)
        true = s.get("scene")
        for expert in experts:
            logits, _ = expert.forward(features[idx:idx + 1])
            pred = argmax_idx(logits[0])
            row = table[expert.name].setdefault(device, [0, 0])
            if true is not None and pred == int(true):
                row[0] += 1
            row[1] += 1
    return table


def write_csv(table: Dict[str, Any], out_csv: str):
    # columns: device, expert, accuracy
    rows = []
    for expert, dmap in table.items():
        for device, (correct, total) in dmap.items():
            acc = correct / total if total > 0 else ""
            rows.append({"device": device, "expert": expert, "accuracy": acc, "correct": correct, "total": total})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["device", "expert", "accuracy", "correct", "total"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--experts", required=False, help='JSON list of expert configs string or path to json')
    args = parser.parse_args()

    if args.experts:
        try:
            expert_cfgs = json.loads(args.experts)
        except Exception:
            expert_cfgs = json.load(open(args.experts, "r", encoding="utf-8"))
    else:
        expert_cfgs = [{"name": "passt"}, {"name": "cnn"}]

    table = compute_table(args.val_manifest, expert_cfgs)
    write_csv(table, args.out_csv)
    print(f"Wrote per-device table to {args.out_csv}")


if __name__ == "__main__":
    main()
