"""
Expert optimization + per-DD ensemble evaluation for a single detector type.

Optimizes K profile-experts for one DD type (7 studies × 50 trials each),
then evaluates the per-DD ensemble (all K experts together) on the eval set.
Designed to be submitted as one SLURM job per detector type (7 jobs total).

Usage:
    python split_pipeline/expert_optimize.py \
        --optuna-storage sqlite:///expert_ensemble_optuna.db \
        --detector-type BNDM \
        --n-streams 10 --base-stream-seed 42 \
        --drift-frequencies 200,400,500,750,1000,1250,1500,2000,2500,3000 \
        --stream-length 8000 --eval-stream-indices 1,4,8 \
        --generators SineClusters,WaveformDrift2,... \
        --n-trials-expert 50 --per-trial-timeout 1200 \
        --n-jobs 7 --output-dir expert_ensemble_results
"""

import os
import sys
import csv
import json
import logging
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.synthetic_f1_multistream_optimize_optuna import (
    _resolve_stream_seeds,
    _default_tolerances,
)
from optimization.expert_ensemble_optuna import (
    StreamProfile,
    DEFAULT_PROFILES,
    DETECTOR_TYPES,
    get_profile_indices,
    optimize_single_detector_expert,
    evaluate_ensemble,
    _append_result_csv,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = ArgumentParser(description="Expert-only optimization for a single DD type")
    ap.add_argument("--optuna-storage", required=True)
    ap.add_argument("--detector-type", required=True, choices=DETECTOR_TYPES)
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
    ap.add_argument("--n-trials-expert", type=int, default=50)
    ap.add_argument("--per-trial-timeout", type=int, default=1200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--profiles", type=str, default=None,
                   help="JSON string defining custom profiles")
    ap.add_argument("--n-jobs", type=int, default=7,
                   help="Number of parallel profile optimizations (default: 7 = one per profile)")
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
    stream_seeds = _resolve_stream_seeds(args.stream_seeds, args.base_stream_seed,
                                         args.n_streams)
    if args.tolerances:
        tolerances = [int(t.strip()) for t in args.tolerances.split(',')]
    else:
        tolerances = _default_tolerances(drift_frequencies)

    eval_indices = [int(i.strip()) for i in args.eval_stream_indices.split(',')]

    if args.profiles:
        profile_data = json.loads(args.profiles)
        profiles = [StreamProfile(**p) for p in profile_data]
    else:
        profiles = DEFAULT_PROFILES

    logger.info("=" * 80)
    logger.info(f"Expert optimization: {args.detector_type}")
    logger.info("=" * 80)
    logger.info(f"  Generators: {generators}")
    logger.info(f"  Drift frequencies: {drift_frequencies}")
    logger.info(f"  Stream length: {args.stream_length}")
    logger.info(f"  Eval indices: {eval_indices}")
    logger.info(f"  Profiles: {[p.name for p in profiles]}")
    logger.info(f"  Expert trials: {args.n_trials_expert}")
    logger.info(f"  Per-trial timeout: {args.per_trial_timeout}s")
    logger.info(f"  N jobs: {args.n_jobs}")

    # Get profile indices (filter out eval streams)
    eval_set = set(eval_indices)
    profile_indices = get_profile_indices(profiles, generators, drift_frequencies)
    for profile_name in profile_indices:
        profile_indices[profile_name] = [i for i in profile_indices[profile_name]
                                         if i not in eval_set]
    for profile_name, indices in profile_indices.items():
        logger.info(f"  Profile {profile_name}: {len(indices)} streams (indices: {indices})")

    os.makedirs(args.output_dir, exist_ok=True)
    expert_csv = os.path.join(args.output_dir, "experts.csv")

    # Build expert tasks: all profiles for this single DD type
    expert_tasks = []
    for profile in profiles:
        expert_tasks.append({
            'optuna_storage': args.optuna_storage,
            'generators': generators,
            'drift_frequencies': drift_frequencies,
            'stream_length': args.stream_length,
            'stream_seeds': stream_seeds,
            'tolerances': tolerances,
            'profile_name': profile.name,
            'profile_indices': profile_indices[profile.name],
            'detector_type': args.detector_type,
            'detector_seed': args.seed,
            'n_trials': args.n_trials_expert,
            'per_trial_timeout': args.per_trial_timeout,
        })

    experts = {}  # profile_name -> result dict
    logger.info(f"Total expert tasks: {len(expert_tasks)} (profiles for {args.detector_type})")

    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {
                pool.submit(optimize_single_detector_expert, **task): task
                for task in expert_tasks
            }
            first_done = True
            for future in as_completed(futures):
                result = future.result()
                if 'error' not in result:
                    experts[result['profile_name']] = result
                    _append_result_csv(expert_csv, result, first_done)
                    first_done = False
                    logger.info(
                        f"Expert {result['profile_name']}_{result['detector_type']}: "
                        f"F1={result['best_trial_value']:.4f} (saved)")
    else:
        first_done = True
        for task in expert_tasks:
            result = optimize_single_detector_expert(**task)
            if 'error' not in result:
                experts[result['profile_name']] = result
                _append_result_csv(expert_csv, result, first_done)
                first_done = False
                logger.info(
                    f"Expert {result['profile_name']}_{result['detector_type']}: "
                    f"F1={result['best_trial_value']:.4f} (saved)")

    logger.info(f"Expert results saved to {expert_csv}")

    # ------------------------------------------------------------------
    # Evaluate per-DD ensemble (all K experts together) on eval set
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"Evaluating per-DD ensemble for {args.detector_type}")
    logger.info("=" * 80)

    expert_configs = list(experts.values())
    if len(expert_configs) == 0:
        logger.warning("No experts available, skipping ensemble evaluation")
        return

    ens_result = evaluate_ensemble(
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=args.stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        eval_indices=eval_indices,
        expert_configs=expert_configs,
        detector_seed=args.seed,
    )
    ens_result['detector_type'] = args.detector_type
    ens_result['ensemble_type'] = 'per_dd'

    phase1_csv = os.path.join(args.output_dir, "phase1_per_dd_ensembles.csv")
    is_first = not os.path.exists(phase1_csv)
    _append_result_csv(phase1_csv, ens_result, is_first)
    logger.info(f"Per-DD ensemble {args.detector_type}: "
                f"macroF1={ens_result['macro_f1']:.4f} (saved to {phase1_csv})")

    logger.info("Expert optimization + evaluation complete.")


if __name__ == "__main__":
    main()
