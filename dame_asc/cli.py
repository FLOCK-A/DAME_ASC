import argparse
from typing import List

from .config import load_config
from .pipeline import run_inference
from .model.placeholder import PlaceholderModel


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DAME-ASC CLI (placeholder)")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML file")
    parser.add_argument("--mode", "-m", choices=["infer", "dryrun"], default="infer")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # Build placeholder model (factory can be added later)
    model = PlaceholderModel(cfg.get("model", {}))

    if args.mode == "dryrun":
        print("Dry run completed. Config loaded and model initialized.")
        return 0

    # Run inference pipeline
    result = run_inference(cfg, model)
    print(f"Inference completed. Processed {result.get('n_samples', 0)} samples.")
    return 0

