from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dame_asc.augment.dcdir_bank import MelEQBank
from dame_asc.features import prepare_features
from dame_asc.models.factory import build_expert, build_fusion
from dame_asc.optim.adamw import AdamW


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    label_smoothing: float = 0.0,
) -> Tuple[float, np.ndarray]:
    num_classes = logits.shape[1]
    probs = softmax(logits)
    if label_smoothing > 0.0:
        eps = float(label_smoothing)
        y = np.full_like(probs, eps / float(num_classes))
        y[np.arange(len(targets)), targets] = 1.0 - eps + eps / float(num_classes)
    else:
        y = np.zeros_like(probs)
        y[np.arange(len(targets)), targets] = 1.0
    loss = -np.sum(y * np.log(np.clip(probs, 1e-12, 1.0))) / float(len(targets))
    dlogits = (probs - y) / float(len(targets))
    return float(loss), dlogits


def get_targets(samples: List[Dict[str, Any]], num_classes: int) -> np.ndarray:
    targets = []
    for s in samples:
        true_raw = s.get("scene", 0)
        try:
            true_int = int(true_raw)
        except Exception:
            true_int = 0
        if true_int < 0 or true_int >= num_classes:
            true_int = int(true_int % num_classes)
        targets.append(true_int)
    return np.array(targets, dtype=np.int64)


def build_modules(cfg: Dict[str, Any]):
    expert_cfgs = cfg.get("model", {}).get("experts", [{"name": "passt"}, {"name": "cnn"}])
    input_cfg = cfg.get("input", {}) or {}
    input_dim = int(input_cfg.get("n_mels", 128))
    experts = []
    for e in expert_cfgs:
        if isinstance(e, dict):
            e_cfg = dict(e)
            e_cfg.setdefault("input_dim", input_dim)
        else:
            e_cfg = {"input_dim": input_dim}
        name = e.get("name") if isinstance(e, dict) else e
        experts.append(build_expert(name, e_cfg))
    fusion_cfg = cfg.get("model", {}).get("fusion", {"name": "dcf"})
    if fusion_cfg:
        fusion_cfg = dict(fusion_cfg)
        fusion_cfg.setdefault("expert_names", [e.name for e in experts])
    fusion = build_fusion(fusion_cfg.get("name", "dcf"), fusion_cfg) if fusion_cfg else None
    augment_cfg = cfg.get("augment", {}) or {}
    dcdir_cfg = augment_cfg.get("dcdir", {}) if isinstance(augment_cfg, dict) else {}
    dcdir = None
    if dcdir_cfg.get("enable", False) and dcdir_cfg.get("mode") == "mel_eq_bank":
        input_cfg = cfg.get("input", {}) or {}
        dcdir = MelEQBank(
            bank_size=int(dcdir_cfg.get("bank_size", 8)),
            n_mels=int(input_cfg.get("n_mels", 128)),
            max_db=float(dcdir_cfg.get("max_db", 6.0)),
            smooth_kernel=int(dcdir_cfg.get("smooth_kernel", 9)),
            embed_dim=int(dcdir_cfg.get("embed_dim", 16)),
            num_devices=int(dcdir_cfg.get("num_devices", 16)),
        )
    return experts, fusion, dcdir


def collect_params(modules: List[Any]) -> List[np.ndarray]:
    params = []
    for mod in modules:
        if mod is None:
            continue
        params.extend(mod.parameters())
    return params


def collect_grads(modules: List[Any]) -> List[np.ndarray]:
    grads = []
    for mod in modules:
        if mod is None:
            continue
        grads.extend(mod.gradients())
    return grads


def zero_grad(modules: List[Any]):
    for mod in modules:
        if mod is None:
            continue
        mod.zero_grad()


def save_checkpoint(
    path: str,
    experts: List[Any],
    fusion: Optional[Any],
    dcdir: Optional[Any],
    meta: Dict[str, Any],
):
    payload: Dict[str, Any] = {"meta": np.array([json.dumps(meta)], dtype=object)}
    for idx, expert in enumerate(experts):
        state = expert.state_dict()
        for key, value in state.items():
            payload[f"experts.{idx}.{key}"] = value
    if fusion is not None:
        state = fusion.state_dict()
        for key, value in state.items():
            payload[f"fusion.{key}"] = value
    if dcdir is not None:
        state = dcdir.state_dict()
        for key, value in state.items():
            payload[f"dcdir.{key}"] = value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, **payload)


def load_checkpoint(path: str, experts: List[Any], fusion: Optional[Any], dcdir: Optional[Any]):
    data = np.load(path, allow_pickle=True)
    for idx, expert in enumerate(experts):
        state = {}
        prefix = f"experts.{idx}."
        for key in data.files:
            if key.startswith(prefix):
                state[key[len(prefix):]] = data[key]
        if state:
            expert.load_state_dict(state)
    if fusion is not None:
        state = {}
        prefix = "fusion."
        for key in data.files:
            if key.startswith(prefix):
                state[key[len(prefix):]] = data[key]
        if state:
            fusion.load_state_dict(state)
    if dcdir is not None:
        state = {}
        prefix = "dcdir."
        for key in data.files:
            if key.startswith(prefix):
                state[key[len(prefix):]] = data[key]
        if state:
            dcdir.load_state_dict(state)


def train_one_epoch(
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    experts: List[Any],
    fusion: Any,
    dcdir: Optional[Any],
    optimizer: AdamW,
    batch_size: int,
    freeze_experts: bool = False,
    freeze_fusion: bool = False,
    freeze_dcdir: bool = False,
):
    rng = np.random.RandomState(0)
    num_classes = int(cfg.get("model", {}).get("num_classes", 3))
    loss_cfg = cfg.get("loss", {}) or {}
    ce_cfg = loss_cfg.get("ce", {}) or {}
    cons_cfg = loss_cfg.get("consistency", {}) or {}
    reg_cfg = loss_cfg.get("reg", {}) or {}
    label_smoothing = float(ce_cfg.get("label_smoothing", 0.0))
    cons_enable = bool(cons_cfg.get("enable", False))
    cons_weight = float(cons_cfg.get("weight", 0.0))
    cons_type = cons_cfg.get("type", "mse")
    gate_entropy_weight = float(reg_cfg.get("gate_entropy_weight", 0.0))
    dcdir_l2_weight = float(reg_cfg.get("dcdir_l2_weight", 0.0))

    total_loss = 0.0
    num_batches = 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        if not batch:
            continue
        zero_grad([*experts, fusion, dcdir])
        features, dcdir_caches, dcdir_applied = prepare_features(batch, cfg, dcdir, training=True, rng=rng)
        expert_logits = []
        expert_caches = []
        for expert in experts:
            logits, cache = expert.forward(features)
            expert_logits.append(logits)
            expert_caches.append(cache)
        expert_logits = np.stack(expert_logits, axis=0)
        device_ids = np.array(
            [-1 if s.get("device", -1) is None else int(s.get("device", -1)) for s in batch],
            dtype=np.int32,
        )
        if fusion is not None:
            fused_logits, fusion_caches = fusion.forward(expert_logits, device_ids)
        else:
            fused_logits = expert_logits.mean(axis=0)
            fusion_caches = []
        targets = get_targets(batch, num_classes)
        ce_loss, dlogits = cross_entropy_loss(fused_logits, targets, label_smoothing=label_smoothing)
        loss = ce_loss
        if cons_enable and cons_weight > 0.0:
            aug_features, _, _ = prepare_features(
                batch,
                cfg,
                dcdir,
                training=True,
                rng=np.random.RandomState(1 + num_batches),
            )
            aug_expert_logits = []
            for expert in experts:
                aug_logits, _ = expert.forward(aug_features)
                aug_expert_logits.append(aug_logits)
            aug_expert_logits = np.stack(aug_expert_logits, axis=0)
            if fusion is not None:
                aug_fused_logits, _ = fusion.forward(aug_expert_logits, device_ids)
            else:
                aug_fused_logits = aug_expert_logits.mean(axis=0)
            if cons_type == "kl":
                p = softmax(fused_logits)
                q = softmax(aug_fused_logits)
                cons = float(np.mean(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(q, 1e-12, 1.0)))))
                dlogits += cons_weight * (p - q) / float(len(batch))
            else:
                diff = fused_logits - aug_fused_logits
                cons = float(np.mean(diff ** 2))
                dlogits += cons_weight * (2.0 * diff) / float(len(batch))
            loss += cons_weight * cons
        if fusion is not None and gate_entropy_weight > 0.0:
            entropies = []
            for cache in fusion_caches:
                pi = cache["pi"]
                entropies.append(-np.sum(pi * np.log(np.clip(pi, 1e-12, 1.0))))
            loss += gate_entropy_weight * float(np.mean(entropies))
            fusion.add_gate_entropy_grad(fusion_caches, gate_entropy_weight / float(len(fusion_caches)))
        if dcdir is not None and dcdir_l2_weight > 0.0:
            loss += dcdir_l2_weight * float(np.mean(dcdir.prototypes ** 2))
            loss += dcdir_l2_weight * float(np.mean(dcdir.proj_w ** 2))
            loss += dcdir_l2_weight * float(np.mean(dcdir.proj_b ** 2))
            proto_numel = float(dcdir.prototypes.size)
            proj_w_numel = float(dcdir.proj_w.size)
            proj_b_numel = float(dcdir.proj_b.size)
            dcdir.grad_prototypes += (2.0 * dcdir_l2_weight / max(1.0, proto_numel)) * dcdir.prototypes
            dcdir.grad_proj_w += (2.0 * dcdir_l2_weight / max(1.0, proj_w_numel)) * dcdir.proj_w
            dcdir.grad_proj_b += (2.0 * dcdir_l2_weight / max(1.0, proj_b_numel)) * dcdir.proj_b
        if fusion is not None:
            grad_expert_logits = fusion.backward(dlogits, fusion_caches)
        else:
            grad_expert_logits = np.repeat(dlogits[None, :, :], expert_logits.shape[0], axis=0) / float(expert_logits.shape[0])
        grad_features = np.zeros_like(features)
        if not freeze_experts:
            for idx, expert in enumerate(experts):
                grad_features += expert.backward(grad_expert_logits[idx], expert_caches[idx])
        if dcdir is not None and (not freeze_dcdir) and any(dcdir_applied):
            input_cfg = cfg.get("input", {}) or {}
            n_frames = int(input_cfg.get("n_frames", 10))
            grad_mel = (grad_features / float(n_frames)).astype(np.float32)
            for i, cache in enumerate(dcdir_caches):
                if cache is None:
                    continue
                dcdir.backward(np.repeat(grad_mel[i][None, :], n_frames, axis=0), cache)
        params = []
        grads = []
        if not freeze_experts:
            params.extend(collect_params(experts))
            grads.extend(collect_grads(experts))
        if not freeze_fusion:
            params.extend(collect_params([fusion]))
            grads.extend(collect_grads([fusion]))
        if dcdir is not None and not freeze_dcdir:
            params.extend(collect_params([dcdir]))
            grads.extend(collect_grads([dcdir]))
        optimizer.set_params(params)
        optimizer.step(grads)
        total_loss += loss
        num_batches += 1
    return {"loss": total_loss / max(1, num_batches)}
