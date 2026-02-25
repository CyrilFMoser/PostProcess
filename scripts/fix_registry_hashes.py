"""
Recompute registry hashes to include the loss field.
Old runs (config.yaml without a loss key) are treated as CE with defaults.

Usage:
    python scripts/fix_registry_hashes.py <base_checkpoint_dir> [--dry-run]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

DEFAULT_LOSS = {"type": "ce", "label_smoothing": 0.0}


def compute_hash(cfg: DictConfig) -> str:
    m = OmegaConf.to_container(cfg.model, resolve=True)
    m.pop("batch_size", None)
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True) if hasattr(cfg, "loss") else DEFAULT_LOSS
    identity = {
        "model": m,
        "class_subset": int(cfg.class_subset),
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "scheduler": OmegaConf.to_container(cfg.scheduler, resolve=True),
        "lr": float(cfg.training.lr),
        "excluded_classes": sorted(int(c) for c in cfg.training.excluded_classes),
        "run_tag": cfg.get("run_tag", ""),
        "loss": loss_cfg,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]


def fix_registry(base_dir: Path, dry_run: bool):
    changed = 0
    for registry_path in base_dir.rglob("registry.json"):
        registry = json.loads(registry_path.read_text())
        dirty = False
        for entry in registry:
            config_path = Path(entry["checkpoint_dir"]) / "config.yaml"
            if not config_path.exists():
                print(f"  [WARN] no config.yaml in {entry['checkpoint_dir']}, skipping")
                continue
            cfg = OmegaConf.load(config_path)
            correct_hash = compute_hash(cfg)
            if entry["hash"] != correct_hash:
                print(
                    f"  run_{entry['run_id']:04d}  {entry['hash']} -> {correct_hash}"
                    + (" (loss defaulted to CE)" if not hasattr(cfg, "loss") else "")
                )
                if not dry_run:
                    entry["hash"] = correct_hash
                    dirty = True
                changed += 1
        if dirty:
            registry_path.write_text(json.dumps(registry, indent=2))
            print(f"  Saved {registry_path}")

    if changed == 0:
        print("All hashes are already correct.")
    elif dry_run:
        print(f"\n{changed} entries would be updated (dry-run, nothing written).")
    else:
        print(f"\n{changed} entries updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_checkpoint_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.base_checkpoint_dir.exists():
        print(f"Directory not found: {args.base_checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    fix_registry(args.base_checkpoint_dir, args.dry_run)
