"""Compute per-expert per-device accuracy table from a manifest.

Usage (module script):
  python -m tools.build_device_table --manifest data/manifests/train.json

This script will load manifest, run each expert's predict on samples,
compare argmax(logits) to sample['scene'] and print a simple table.
"""
import argparse
import json
from typing import Dict, Any

from dame_asc.data.loader import DataLoader
from dame_asc.models.factory import build_expert


def argmax_idx(logits):
    return int(max(range(len(logits)), key=lambda i: logits[i]))


def compute_table(manifest_path: str, experts: Dict[str, Dict[str, Any]]):
    loader = DataLoader(manifest_path)
    samples = loader.load_manifest()
    # initialize counts
    table = {ename: {} for ename in experts.keys()}  # expert -> device -> (correct, total)

    for s in samples:
        device = int(s.get("device", -1) or -1)
        true = s.get("scene")
        for ename in experts.keys():
            expert = build_expert(ename, experts[ename])
            out = expert.predict(s)
            pred = argmax_idx(out["logits"])
            row = table[ename].setdefault(device, [0, 0])
            if true is not None and pred == int(true):
                row[0] += 1
            row[1] += 1
    # build accuracy
    acc = {e: {d: (v[0] / v[1] if v[1] > 0 else None) for d, v in table[e].items()} for e in table}
    return acc


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/train.json")
    args = parser.parse_args(argv)

    # Example experts config (could be read from config)
    experts = {"passt": {}, "cnn": {}}
    acc = compute_table(args.manifest, experts)
    print(json.dumps(acc, indent=2))


if __name__ == "__main__":
    main()
