"""Minimal placeholder training script for Stage-1 (general).

This script runs a synthetic, deterministic training loop for smoke testing
and demonstrates how losses (CE, consistency, reg) are computed.
"""
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
from typing import Dict, Any, List

import numpy as np

from dame_asc.config import load_config
from dame_asc.data.loader import DataLoader
from dame_asc.models.factory import build_expert
from dame_asc.losses.ce import batch_ce
from dame_asc.losses.consistency import consistency_loss
from dame_asc.losses.reg import gate_entropy, dcdir_l2_norm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_manifest", required=False)
    p.add_argument("--val_manifest", required=False)
    p.add_argument("--workdir", required=False, default="runs/stage1")
    p.add_argument("--epochs", type=int, required=False)
    return p.parse_args()


def one_epoch(samples: List[Dict[str, Any]], experts: List, cfg: Dict[str, Any]) -> Dict[str, float]:
    # Simple per-sample loop computing losses
    ce_ls = []
    cons_ls = []
    gate_ent_ls = []
    dcdir_norm_ls = []

    loss_cfg = cfg.get("loss", {}) or {}
    ce_cfg = loss_cfg.get("ce", {}) or {}
    cons_cfg = loss_cfg.get("consistency", {}) or {}
    reg_cfg = loss_cfg.get("reg", {}) or {}

    label_smoothing = float(ce_cfg.get("label_smoothing", 0.0))
    cons_enable = bool(cons_cfg.get("enable", False))
    cons_weight = float(cons_cfg.get("weight", 0.0))
    cons_type = cons_cfg.get("type", "kl")
    gate_entropy_weight = float(reg_cfg.get("gate_entropy_weight", 0.0))
    dcdir_l2_weight = float(reg_cfg.get("dcdir_l2_weight", 0.0))

    num_classes = int(cfg.get("model", {}).get("num_classes", 3))

    for s in samples:
        # collect expert logits
        expert_outputs = []
        for ex in experts:
            out = ex.predict(s)
            expert_outputs.append(out)

        # For CE, fuse via simple average logits as placeholder
        logits_mat = np.vstack([np.array(o["logits"]) for o in expert_outputs])
        avg_logits = logits_mat.mean(axis=0)
        true_raw = s.get("scene", None)
        if true_raw is None:
            true = 0
        else:
            try:
                true_int = int(true_raw)
            except Exception:
                true_int = 0
            if true_int < 0 or true_int >= num_classes:
                print(f"[warning] sample {s.get('id')} scene={true_int} out of range 0..{num_classes-1}, mapping via modulo")
                true = int(true_int % num_classes)
            else:
                true = true_int
        ce = batch_ce(avg_logits[np.newaxis, :], np.array([int(true)]), label_smoothing=label_smoothing)
        ce_ls.append(ce)

        if cons_enable:
            # simulate augmented sample by applying a tiny perturbation to logits
            perturbed = avg_logits + 0.01
            cons = consistency_loss(avg_logits, perturbed, loss_type=cons_type)
            cons_ls.append(cons * cons_weight)

        if gate_entropy_weight > 0.0:
            # simulate gating distribution uniform
            K = len(experts)
            pi = np.ones(K) / float(K)
            gate_ent_ls.append(gate_entropy(pi) * gate_entropy_weight)

        if dcdir_l2_weight > 0.0:
            # simulate prototypes list
            prototypes = [np.zeros(10) for _ in range(4)]
            dcdir_norm_ls.append(dcdir_l2_norm(prototypes) * dcdir_l2_weight)

    total_loss = float(np.mean(ce_ls) + sum(cons_ls) + sum(gate_ent_ls) + sum(dcdir_norm_ls))
    return {"ce": float(np.mean(ce_ls)), "cons": float(sum(cons_ls)), "reg_gate": float(sum(gate_ent_ls)), "reg_dcdir": float(sum(dcdir_norm_ls)), "total": total_loss}


def save_ckpt(workdir: str, epoch: int, metrics: Dict[str, float]):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    ckpt = {"epoch": epoch, "metrics": metrics}
    with open(os.path.join(workdir, "best.ckpt"), "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    workdir = args.workdir
    epochs = int(args.epochs or cfg.get("train", {}).get("epochs", 1))

    # Data
    manifest = args.train_manifest or cfg.get("dataset", {}).get("manifest")
    loader = DataLoader(manifest)
    samples = loader.generate_synthetic(8) if (not manifest) else loader.load_manifest()

    # Build experts
    expert_cfgs = cfg.get("model", {}).get("experts", [{"name": "passt"}, {"name": "cnn"}])
    experts = []
    for e in expert_cfgs:
        name = e.get("name") if isinstance(e, dict) else e
        experts.append(build_expert(name, e if isinstance(e, dict) else {}))

    # train loop (placeholder)
    best_total = float("inf")
    for ep in range(1, epochs + 1):
        metrics = one_epoch(samples, experts, cfg)
        print(f"Epoch {ep}/{epochs} metrics: {metrics}")
        if metrics["total"] < best_total:
            save_ckpt(workdir, ep, metrics)
            best_total = metrics["total"]


if __name__ == "__main__":
    main()
