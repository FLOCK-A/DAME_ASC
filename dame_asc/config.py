import json
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML or JSON config and do basic validation.

    If the config is YAML and PyYAML is not installed, raise ValueError with an
    actionable message.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        if path.lower().endswith(".json"):
            cfg = json.loads(text)
        else:
            try:
                import yaml  # type: ignore

                cfg = yaml.safe_load(text)
            except Exception as exc:
                raise ValueError(
                    "YAML config requested but PyYAML is not available. "
                    "Install PyYAML or use a JSON config."
                ) from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse config {path}: {exc}") from exc

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping (dict) at top level")

    # Minimal defaults
    cfg.setdefault("model", {})
    cfg.setdefault("inference", {})
    return cfg
