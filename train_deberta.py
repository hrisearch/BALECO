#!/usr/bin/env python3
"""Fine-tune DeBERTa v3-base on DiCo-NLI Track 1 (English): the System-1 fast path.

Trains on the official train split and evaluates on the dev split with the
official scorer (Weighted F1, SoftCons, HardCons). The training loop, data
handling and scorer integration are reused from the task's starter kit;
hyperparameters follow its default configuration.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs"


def default_task_repo() -> str:
    """Path to a checkout of the official task repository (data, starter kit, scorer)."""
    env = os.environ.get("DICONLI_REPO")
    if env:
        return env
    return str(SCRIPT_DIR.parent / "SemEval-2027-Task-2-DiCo-NLI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_repo", type=str, default=default_task_repo(),
                        help="official DiCo-NLI repository (data, starter kit, scorer); "
                             "also settable via the DICONLI_REPO environment variable")
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--train_path", type=str, default=None,
                        help="default: <task_repo>/final_data/train/dico_nli_train_track1_participant_labeled.csv")
    parser.add_argument("--dev_path", type=str, default=None,
                        help="default: <task_repo>/final_data/dev/dico_nli_dev_track1_participant_labeled.csv")
    parser.add_argument("--reference_path", type=str, default=None,
                        help="default: <task_repo>/final_data/dev/dico_nli_dev_track1_reference.csv")
    parser.add_argument("--save_path", type=str, default=None,
                        help="default: outputs/models/deberta_v3_base_track1")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="default: outputs/results/deberta_v3_base_track1")
    # Starter-kit default configuration.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_pcrt", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_repo = Path(args.task_repo)
    if args.train_path is None:
        args.train_path = str(task_repo / "final_data/train/dico_nli_train_track1_participant_labeled.csv")
    if args.dev_path is None:
        args.dev_path = str(task_repo / "final_data/dev/dico_nli_dev_track1_participant_labeled.csv")
    if args.reference_path is None:
        args.reference_path = str(task_repo / "final_data/dev/dico_nli_dev_track1_reference.csv")
    if args.save_path is None:
        args.save_path = str(OUTPUT_ROOT / "models/deberta_v3_base_track1")
    if args.results_dir is None:
        args.results_dir = str(OUTPUT_ROOT / "results/deberta_v3_base_track1")

    # Reuse the starter-kit package (its `src.*` imports resolve inside it) and
    # the official scorer package at the task repository root.
    sys.path.insert(0, str(task_repo / "starter_kit"))
    sys.path.insert(0, str(task_repo))
    from src.trainer.trainer import Trainer

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "model_path": args.model_path,
        "tokenizer_path": args.model_path,
        "do_train": True,
        "train_path": args.train_path,
        "do_dev": True,
        "dev_path": args.dev_path,
        "do_test": False,
        "test_path": None,
        "seed": args.seed,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "warmup_pcrt": args.warmup_pcrt,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "wd": args.wd,
        "num_epochs": args.num_epochs,
        "save_path": args.save_path,
        "epoch_to_stop": None,
        "results_dir": str(results_dir),
        "reference_path": args.reference_path,
        "optuna": None,
    }

    Trainer(params=params)

    # The trainer scores every per-epoch dev prediction file with the official
    # scorer; report the final one.
    score_dirs = sorted(results_dir.glob("predictions_*_official_scores"))
    if score_dirs:
        final = score_dirs[-1]
        print(f"\nFinal official scores directory: {final}")
        scores_file = final / "scores.json"
        if scores_file.exists():
            report = json.loads(scores_file.read_text())
            summary = {key: report[key] for key in ("weighted_f1", "soft_cons", "hard_cons")}
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
