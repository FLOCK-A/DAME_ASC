from typing import Dict, Any

from .experts.passt_expert import PasstExpert
from .experts.cnn_expert import CnnExpert
from .experts.beats_expert import BeatsExpert
from .fusion.dcf import DCFusion


def build_expert(name: str, cfg: Dict[str, Any] | None = None):
    cfg = cfg or {}
    if name == "passt":
        return PasstExpert(cfg)
    if name == "cnn":
        return CnnExpert(cfg)
    if name == "beats":
        return BeatsExpert(cfg)
    raise ValueError(f"Unknown expert {name}")


def build_fusion(name: str, cfg: Dict[str, Any] | None = None):
    cfg = cfg or {}
    if name == "dcf":
        return DCFusion(cfg)
    raise ValueError(f"Unknown fusion {name}")
