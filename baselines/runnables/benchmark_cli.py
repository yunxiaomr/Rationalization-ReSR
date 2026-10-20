from __future__ import annotations

import argparse
from pathlib import Path


def parse_runner_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain each seed even if checkpoints already exist.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit random seeds. Defaults to the five main-experiment seeds.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the configured maximum epochs. Useful for smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the configured batch size.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="Lightning limit_train_batches override for smoke tests.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Lightning limit_val_batches override for smoke tests.",
    )
    parser.add_argument(
        "--limit-test-batches",
        type=float,
        default=None,
        help="Lightning limit_test_batches override for smoke tests.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    return parser.parse_args()


def resolve_selected_seeds(config_seeds, requested_seeds):
    if requested_seeds is None:
        return list(config_seeds)

    selected = list(requested_seeds)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Provide one or more distinct random seeds.")
    return selected


def list_checkpoints(seed_ckpt_path: Path):
    return sorted(
        seed_ckpt_path.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def run_path(path: Path, args) -> Path:
    fields = (
        ("max_epochs", "epochs"),
        ("batch_size", "batch"),
        ("limit_train_batches", "train"),
        ("limit_val_batches", "val"),
        ("limit_test_batches", "test"),
    )
    tags = []
    for field, label in fields:
        value = getattr(args, field, None)
        if value is not None:
            tag = str(value).replace(".", "p").replace("-", "m")
            tags.append(f"{label}{tag}")
    return path.with_name(f"{path.name}__{'_'.join(tags)}") if tags else path
