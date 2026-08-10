#!/usr/bin/env python3
"""
Cross-dataset evaluation wrapper.

Runs per-dataset hyperparameter optimization for all single drift detectors
and the MoPEDDS ensemble using Optuna, then evaluates the optimized
configurations on the same dataset via the ML pipeline.

This script wraps:
  - optimization/single_dd_optimize_optuna.py  (single-DD optimization)
  - optimization/mopedds_optimize_optuna.py     (MoPEDDS ensemble optimization)

By default, optimization is run for all supported datasets.  A subset can be
selected via --datasets.  All arguments except --datasets are forwarded to
the underlying optimization scripts.

Usage:
    python run_cross_dataset_evaluation.py [--datasets DS1 DS2 ...]
        [--n_trials N] [--n_jobs J] [--timeout T] [--output_dir DIR]

Examples:
    # Optimize on all datasets with defaults
    python run_cross_dataset_evaluation.py

    # Optimize on Electricity and PokerHand with 500 trials, 4 parallel jobs
    python run_cross_dataset_evaluation.py --datasets Electricity PokerHand --n_trials 500 --n_jobs 4
"""

import os
import sys
import subprocess
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASETS = [
    "Electricity",
    "GasSensor",
    "PokerHand",
    "RialtoBridgeTimelapse",
]

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_optimization(script, dataset, extra_args):
    """Run an optimization script as a subprocess for a single dataset."""
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, script),
        "--dataset", dataset,
    ] + extra_args
    logger.info("Starting %s for dataset=%s", script, dataset)
    logger.info("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        logger.error("%s failed for dataset=%s (exit code %d)",
                     script, dataset, result.returncode)
        return False
    logger.info("Completed %s for dataset=%s", script, dataset)
    return True


def main():
    parser = ArgumentParser(
        description="Cross-dataset evaluation: per-dataset Optuna optimization "
                    "of single DDs and MoPEDDS ensemble.")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=DEFAULT_DATASETS,
                        help="Datasets to optimize and evaluate on "
                             "(default: all supported datasets)")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of Optuna trials per detector/ensemble")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel Optuna jobs")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds per detector/ensemble")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for CSV result files")
    parser.add_argument("--skip_single_dd", action="store_true",
                        help="Skip single-DD optimization (run only MoPEDDS)")
    parser.add_argument("--skip_mopedds", action="store_true",
                        help="Skip MoPEDDS optimization (run only single DDs)")
    args = parser.parse_args()

    extra_args = [
        "--n_trials", str(args.n_trials),
        "--n_jobs", str(args.n_jobs),
        "--output_dir", args.output_dir,
    ]
    if args.timeout is not None:
        extra_args += ["--timeout", str(args.timeout)]

    failed = []
    for dataset in args.datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", dataset)
        logger.info("=" * 60)

        if not args.skip_single_dd:
            ok = run_optimization(
                "optimization/single_dd_optimize_optuna.py",
                dataset, extra_args)
            if not ok:
                failed.append(f"single_dd:{dataset}")

        if not args.skip_mopedds:
            ok = run_optimization(
                "optimization/mopedds_optimize_optuna.py",
                dataset, extra_args)
            if not ok:
                failed.append(f"mopedds:{dataset}")

    if failed:
        logger.error("Failed optimizations: %s", ", ".join(failed))
        sys.exit(1)

    logger.info("All optimizations completed successfully.")


if __name__ == "__main__":
    main()
