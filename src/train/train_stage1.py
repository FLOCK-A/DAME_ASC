"""Stage-1 (general) training script with numpy-based training loop."""
import sys
from pathlib import Path as _Path
# Ensure project root is on sys.path so 'dame_asc' package can be imported when running this script
_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

from dame_asc.config import load_config
from dame_asc.data.loader import DataLoader
from dame_asc.optim.adamw import AdamW
from src.train.train_utils import build_modules, save_checkpoint, train_one_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_manifest", required=False)
    p.add_argument("--val_manifest", required=False)
    p.add_argument("--workdir", required=False, default="runs/stage1")
    p.add_argument("--epochs", type=int, required=False)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    workdir = args.workdir
    epochs = int(args.epochs or cfg.get("train", {}).get("epochs", 1))

    # Data
    manifest = args.train_manifest or cfg.get("dataset", {}).get("manifest")
    loader = DataLoader(manifest)
    samples = loader.generate_synthetic(8) if (not manifest) else loader.load_manifest()

    experts, fusion, dcdir = build_modules(cfg)
    train_cfg = cfg.get("train", {}) or {}
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    freeze_cfg = train_cfg.get("freeze", {}) or {}
    freeze_experts = bool(freeze_cfg.get("experts", False))
    freeze_fusion = bool(freeze_cfg.get("fusion", False))
    freeze_dcdir = bool(freeze_cfg.get("dcdir", False))
    optimizer = AdamW([], lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        metrics = train_one_epoch(
            samples,
            cfg,
            experts,
            fusion,
            dcdir,
            optimizer,
            batch_size,
            freeze_experts=freeze_experts,
            freeze_fusion=freeze_fusion,
            freeze_dcdir=freeze_dcdir,
        )
        gate_stats = metrics.pop("gate_stats", None)
        dcdir_eq = metrics.pop("dcdir_eq", None)
        Path(workdir).mkdir(parents=True, exist_ok=True)
        if gate_stats:
            gate_path = os.path.join(workdir, f"gating_stats_epoch_{ep}.json")
            with open(gate_path, "w", encoding="utf-8") as f:
                json.dump(gate_stats, f, ensure_ascii=False, indent=2)
        if dcdir_eq:
            eq_path = os.path.join(workdir, f"dcdir_eq_epoch_{ep}.json")
            with open(eq_path, "w", encoding="utf-8") as f:
                json.dump(dcdir_eq, f, ensure_ascii=False, indent=2)
        print(f"Epoch {ep}/{epochs} metrics: {metrics}")
        if metrics["loss"] < best_loss:
            meta = {"epoch": ep, "metrics": metrics, "stage": "general"}
            save_checkpoint(os.path.join(workdir, "best.ckpt"), experts, fusion, dcdir, meta)
            best_loss = metrics["loss"]


if __name__ == "__main__":
    main()
