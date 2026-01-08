"""Stage-2 device-specific fine-tuning with numpy-based training loop."""
import argparse
import json
import os
from pathlib import Path

from dame_asc.config import load_config
from dame_asc.data.loader import DataLoader
from dame_asc.optim.adamw import AdamW
from src.train.train_utils import build_modules, load_checkpoint, save_checkpoint, train_one_epoch


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

    experts, fusion, dcdir = build_modules(cfg)
    if args.init_ckpt:
        load_checkpoint(args.init_ckpt, experts, fusion, dcdir)

    train_cfg = cfg.get("train", {}) or {}
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    freeze_cfg = train_cfg.get("freeze", {}) or {}
    freeze_experts = bool(freeze_cfg.get("experts", True))
    freeze_fusion = bool(freeze_cfg.get("fusion", False))
    freeze_dcdir = bool(freeze_cfg.get("dcdir", True))
    optimizer = AdamW([], lr=lr, weight_decay=weight_decay)

    def train_for_device(device_id: int, subset: list, out_dir: str):
        best_loss = float("inf")
        epochs = int(train_cfg.get("epochs", 1))
        for ep in range(1, epochs + 1):
            metrics = train_one_epoch(
                subset,
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
            if gate_stats:
                gate_path = os.path.join(out_dir, f"gating_stats_epoch_{ep}.json")
                with open(gate_path, "w", encoding="utf-8") as f:
                    json.dump(gate_stats, f, ensure_ascii=False, indent=2)
            if dcdir_eq:
                eq_path = os.path.join(out_dir, f"dcdir_eq_epoch_{ep}.json")
                with open(eq_path, "w", encoding="utf-8") as f:
                    json.dump(dcdir_eq, f, ensure_ascii=False, indent=2)
            print(f"[device {device_id}] Epoch {ep}/{epochs} metrics: {metrics}")
            if metrics["loss"] < best_loss:
                meta = {"epoch": ep, "metrics": metrics, "stage": "device_specific", "device_id": device_id}
                save_checkpoint(os.path.join(out_dir, "best.ckpt"), experts, fusion, dcdir, meta)
                best_loss = metrics["loss"]

    Path(workdir).mkdir(parents=True, exist_ok=True)
    if args.all_devices:
        device_ids = sorted(
            {
                (-1 if s.get("device", -1) is None else int(s.get("device", -1)))
                for s in samples
                if (-1 if s.get("device", -1) is None else int(s.get("device", -1))) >= 0
            }
        )
        for device_id in device_ids:
            if args.init_ckpt:
                load_checkpoint(args.init_ckpt, experts, fusion, dcdir)
            subset = [
                s
                for s in samples
                if (-1 if s.get("device", -1) is None else int(s.get("device", -1))) == device_id
            ]
            if not subset:
                continue
            device_dir = os.path.join(workdir, f"device_{device_id}")
            Path(device_dir).mkdir(parents=True, exist_ok=True)
            train_for_device(device_id, subset, device_dir)
    else:
        device_id = int(args.device_id) if args.device_id is not None else -1
        if device_id >= 0:
            subset = [
                s
                for s in samples
                if (-1 if s.get("device", -1) is None else int(s.get("device", -1))) == device_id
            ]
        else:
            subset = samples
        train_for_device(device_id, subset, workdir)


if __name__ == "__main__":
    main()

