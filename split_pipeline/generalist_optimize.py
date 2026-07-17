"""
Generalist optimization + evaluation for a single detector type.

Phase 1: Optimize generalist detector on all training streams.
Phase 1b: Evaluate best config as a single detector (no MOPEDDS) on eval set.
Designed to be submitted as one SLURM job per detector type (7 jobs total).

Usage:
    python split_pipeline/generalist_optimize.py \
        --optuna-storage sqlite:///split_pipeline/split_pipeline_optuna.db \
        --n-streams 10 --base-stream-seed 42 \
        --drift-frequencies 200,400,500,750,1000,1250,1500,2000,2500,3000 \
        --stream-length 8000 --eval-stream-indices 1,4,8 \
        --generators SineClusters,WaveformDrift2,... \
        --detector-type IBDD --n-trials 700 --n-trials-deployment 100 \
        --output-dir split_pipeline/results
"""

import os
import sys
import csv
import json
import logging
import optuna
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.synthetic_f1_multistream_optimize_optuna import (
    _resolve_stream_seeds,
    _default_tolerances,
)
from optimization.expert_ensemble_optuna import (
    StreamProfile,
    DEFAULT_PROFILES,
    DETECTOR_TYPES,
    optimize_generalist_detector,
    _append_result_csv,
)
from optimization.single_dd_optuna import evaluate_detector

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = ArgumentParser(description="Generalist optimization for a single DD type")
    ap.add_argument("--optuna-storage", required=True)
    ap.add_argument("--n-streams", type=int, required=True)
    ap.add_argument("--base-stream-seed", type=int, required=True)
    ap.add_argument("--drift-frequencies", type=str, required=True)
    ap.add_argument("--stream-length", type=int, required=True)
    ap.add_argument("--stream-seeds", type=str, default=None)
    ap.add_argument("--tolerances", type=str, default=None)
    ap.add_argument("--eval-stream-indices", type=str, required=True)
    ap.add_argument("--generators", type=str, default=None)
    ap.add_argument("--generator", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--detector-type", required=True, choices=DETECTOR_TYPES)
    ap.add_argument("--n-trials", type=int, default=800,
                   help="Number of Optuna trials for detector hyperparameter optimization")
    ap.add_argument("--per-trial-timeout", type=int, default=1200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--profiles", type=str, default=None,
                   help="JSON string defining custom profiles")
    ap.add_argument("--load-if-exists", action="store_true",
                   help="Resume an existing Optuna study instead of creating a fresh one")
    args = ap.parse_args()

    # Parse generators
    if args.generators:
        generators = [g.strip() for g in args.generators.split(',')]
        if len(generators) == 1:
            generators = generators * args.n_streams
    elif args.generator:
        generators = [args.generator] * args.n_streams
    else:
        raise ValueError("Must specify --generators or --generator")

    drift_frequencies = [int(f.strip()) for f in args.drift_frequencies.split(',')]
    stream_seeds = _resolve_stream_seeds(args.stream_seeds, args.n_streams,
                                         args.base_stream_seed)
    if args.tolerances:
        tolerances = [int(t.strip()) for t in args.tolerances.split(',')]
    else:
        tolerances = _default_tolerances(drift_frequencies)

    eval_indices = [int(i.strip()) for i in args.eval_stream_indices.split(',')]
    train_indices = [i for i in range(args.n_streams) if i not in eval_indices]

    if args.profiles:
        profile_data = json.loads(args.profiles)
        profiles = [StreamProfile(**p) for p in profile_data]
    else:
        profiles = DEFAULT_PROFILES

    logger.info("=" * 80)
    logger.info(f"Generalist optimization: {args.detector_type}")
    logger.info("=" * 80)
    logger.info(f"  Storage: {args.optuna_storage}")
    logger.info(f"  Train indices: {train_indices}")
    logger.info(f"  N trials: {args.n_trials}")
    logger.info(f"  Per-trial timeout: {args.per_trial_timeout}s")

    os.makedirs(args.output_dir, exist_ok=True)
    generalist_csv = os.path.join(args.output_dir, f"generalists_{args.detector_type.lower()}.csv")

    # ------------------------------------------------------------------
    # Phase 1: Optimize detector hyperparameters
    # ------------------------------------------------------------------
    result = optimize_generalist_detector(
        optuna_storage=args.optuna_storage,
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=args.stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        train_indices=train_indices,
        detector_type=args.detector_type,
        detector_seed=args.seed,
        n_trials=args.n_trials,
        per_trial_timeout=args.per_trial_timeout,
        load_if_exists=args.load_if_exists,
    )

    if 'error' in result:
        logger.error(f"Generalist {args.detector_type} failed: {result.get('error')}")
        return

    _append_result_csv(generalist_csv, result, True)
    logger.info(f"Generalist {args.detector_type}: "
                f"train F1={result['best_trial_value']:.4f} (saved to {generalist_csv})")

    best_params = result['best_params']

    # ------------------------------------------------------------------
    # Phase 1b: Evaluate as single detector (no MOPEDDS) on eval set
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"Phase 1b: Single-detector eval for generalist {args.detector_type}")
    logger.info("=" * 80)

    eval_result = evaluate_detector(
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=args.stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        indices=eval_indices,
        detector_type=args.detector_type,
        detector_params=best_params,
        detector_seed=args.seed,
    )
    eval_result['detector_type'] = args.detector_type
    eval_result['ensemble_type'] = 'generalist_single'

    gen_eval_csv = os.path.join(args.output_dir, f"generalists_eval_{args.detector_type.lower()}.csv")
    _append_result_csv(gen_eval_csv, eval_result, True)
    logger.info(f"Generalist {args.detector_type} single-detector eval: "
                f"macroF1={eval_result['macro_f1']:.4f} (saved to {gen_eval_csv})")

    logger.info("Generalist optimization + evaluation complete.")


if __name__ == "__main__":
    main()
