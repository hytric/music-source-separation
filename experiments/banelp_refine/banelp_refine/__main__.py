"""Preparation and training are distinct, explicit CLI operations."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import emit, load_config, runtime_root
from .contracts import validate_execution_config


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs/pilot.json")
    p.add_argument("--runtime-dir", type=Path, required=True)
    p.add_argument("--baseline-repo", type=Path, required=True)
    p.add_argument("--baseline-runtime", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare-data", "prepare", "rirs", "cache", "preflight", "status"):
        sub.add_parser(command)
    train = sub.add_parser("train")
    train.add_argument("--start-training", action="store_true", help="explicitly permit optimizer updates")
    train.add_argument("--resume", type=Path)
    train.add_argument("--stop-after", type=int, help="stop at this global step without changing the sealed experiment")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "confirmation"), default="confirmation")
    evaluation.add_argument("--include-ensemble", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    validate_execution_config(config)
    root = runtime_root(args.runtime_dir, args.baseline_runtime)
    if args.command == "status":
        emit("status", **{name: (root / path).exists() for name, path in {
            "dataset": "manifests/dataset.json", "rirs": "manifests/rirs.json", "cache": "manifests/cache.json",
            "readiness": "readiness.json", "initial_checkpoint": "initial/step-000000.pt"}.items()})
        return
    if args.command == "train" and not args.start_training:
        raise SystemExit("Refusing optimizer updates: train requires --start-training")
    if args.command in ("prepare-data", "prepare"):
        from .prepare import prepare_data
        prepare_data(root, args.baseline_runtime / "data/musdb18hq.zip", config, args.baseline_runtime)
    if args.command in ("rirs", "prepare"):
        from .rirs import make_rirs
        make_rirs(root, config, args.baseline_repo)
    if args.command in ("cache", "prepare"):
        from .cache import make_cache
        make_cache(root, config, args.baseline_repo, args.baseline_runtime)
    if args.command in ("preflight", "prepare"):
        from .engine import preflight
        preflight(root, config, args.baseline_repo, args.baseline_runtime)
    if args.command == "train":
        from .engine import train
        train(root, config, start_training=args.start_training, resume=args.resume, stop_after=args.stop_after)
    if args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(root, config, args.baseline_repo, args.baseline_runtime, args.checkpoint, args.split, args.include_ensemble)


if __name__ == "__main__":
    main()
