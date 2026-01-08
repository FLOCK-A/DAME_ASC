from pathlib import Path
from dame_asc.cli import main


def _find_config(repo_root: Path) -> Path:
    json_cfg = repo_root / "config.json"
    if json_cfg.exists():
        return json_cfg
    yaml_cfg = repo_root / "config.yaml"
    return yaml_cfg


def test_cli_runs_dryrun():
    repo_root = Path(__file__).resolve().parent.parent
    cfg = _find_config(repo_root)
    assert cfg.exists()

    # Dryrun should return 0
    rc = main(["--config", str(cfg), "--mode", "dryrun"])
    assert rc == 0


def test_infer_produces_output(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    example_cfg = _find_config(repo_root)
    assert example_cfg.exists()
    out_path = tmp_path / "out.json"
    # Run inference (creates inference_results.json by default)
    rc = main(["--config", str(example_cfg), "--mode", "infer"])
    assert rc == 0
    # Check that inference_results.json was created in repo root
    out_file = Path.cwd() / "inference_results.json"
    assert out_file.exists()
