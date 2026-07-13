"""
Evaluate expert ensembles and generalists from Optuna DB.

Loads best trials from the Optuna DB and runs:
  Phase 1: Per-DD ensembles (K experts of same DD type)
  Phase 2: Cross-DD best experts (best expert per profile across all DD types)
  Phase 3: Generalist single-detector evaluation on eval set

This script does NO optimization. It should be submitted after all
expert and generalist optimization jobs have completed.

Usage:
    python optimization/evaluate_ensembles.py \
        --optuna-storage sqlite:///expert_ensemble_optuna.db \
        --n-streams 10 --base-stream-seed 42 \
        --drift-frequencies 200,400,500,750,1000,1250,1500,2000,2500,3000 \
        --stream-length 8000 --eval-stream-indices 1,4,8 \
        --generators SineClusters,WaveformDrift2,... \
        --n-trials-expert 50 --output-dir expert_ensemble_results
"""

import os
import sys
import csv
import json
import logging
from argparse import ArgumentParser

import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.synthetic_f1_multistream_optimize_optuna import (
    _resolve_stream_seeds,
    _default_tolerances,
)
from optimization.expert_ensemble_optuna import (
    StreamProfile,
    DEFAULT_PROFILES,
    DETECTOR_TYPES,
    evaluate_ensemble,
    _append_result_csv,
    _rewrite_csv,
)
from optimization.evaluate_from_optuna import (
    load_expert_from_study,
    load_generalist_from_study,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = ArgumentParser(description="Evaluate expert ensembles and generalists from Optuna DB")
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
    ap.add_argument("--n-trials-expert", type=int, default=50)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--profiles", type=str, default=None)
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
    logger.info("Evaluate ensembles from Optuna DB")
    logger.info("=" * 80)
    logger.info(f"  Storage: {args.optuna_storage}")
    logger.info(f"  Eval indices: {eval_indices}")
    logger.info(f"  Profiles: {[p.name for p in profiles]}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load experts from DB
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Loading experts from Optuna DB")
    logger.info("=" * 80)

    all_experts = {}
    for detector_type in DETECTOR_TYPES:
        for profile in profiles:
            key = (profile.name, detector_type)
            result = load_expert_from_study(
                args.optuna_storage, profile.name, detector_type)
            if result is not None:
                all_experts[key] = result
                logger.info(
                    f"  Loaded expert {profile.name}_{detector_type}: "
                    f"F1={result['best_trial_value']:.4f} "
                    f"({result['n_completed_trials']} trials)")
            else:
                logger.warning(
                    f"  Expert {profile.name}_{detector_type} not found in DB")

    # Save experts CSV
    expert_csv = os.path.join(args.output_dir, "experts.csv")
    expert_results = list(all_experts.values())
    _rewrite_csv(expert_csv, expert_results)
    logger.info(f"Experts CSV saved to {expert_csv} ({len(expert_results)} experts)")

    # ------------------------------------------------------------------
    # Load generalists from DB
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Loading generalists from Optuna DB")
    logger.info("=" * 80)

    all_generalists = {}
    for detector_type in DETECTOR_TYPES:
        result = load_generalist_from_study(
            args.optuna_storage, detector_type)
        if result is not None:
            all_generalists[detector_type] = result
            logger.info(
                f"  Loaded generalist {detector_type}: "
                f"F1={result['best_trial_value']:.4f} "
                f"({result['n_completed_trials']} trials)")
        else:
            logger.warning(
                f"  Generalist {detector_type} not found in DB")

    # Save generalists CSV
    generalist_csv = os.path.join(args.output_dir, "generalists.csv")
    generalist_results = list(all_generalists.values())
    _rewrite_csv(generalist_csv, generalist_results)
    logger.info(f"Generalists CSV saved to {generalist_csv} ({len(generalist_results)} generalists)")

    # ------------------------------------------------------------------
    # Phase 1: Per-DD ensembles
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Phase 1: Per-DD ensembles")
    logger.info("=" * 80)

    phase1_results = []
    phase1_csv = os.path.join(args.output_dir, "phase1_per_dd_ensembles.csv")
    if os.path.exists(phase1_csv):
        os.remove(phase1_csv)

    for detector_type in DETECTOR_TYPES:
        expert_configs = []
        for profile in profiles:
            key = (profile.name, detector_type)
            if key in all_experts:
                expert_configs.append(all_experts[key])

        if len(expert_configs) == 0:
            logger.warning(f"No experts found for {detector_type}, skipping")
            continue

        result = evaluate_ensemble(
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=args.stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            eval_indices=eval_indices,
            expert_configs=expert_configs,
            detector_seed=args.seed,
        )
        result['detector_type'] = detector_type
        result['ensemble_type'] = 'per_dd'
        phase1_results.append(result)
        _append_result_csv(phase1_csv, result, len(phase1_results) == 1)
        logger.info(f"Per-DD ensemble {detector_type}: macroF1={result['macro_f1']:.4f}")

    logger.info(f"Phase 1 results saved to {phase1_csv}")

    # ------------------------------------------------------------------
    # Phase 2: Cross-DD best experts
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Phase 2: Cross-DD best experts")
    logger.info("=" * 80)

    best_experts = []
    for profile in profiles:
        best_f1 = -1
        best_config = None
        best_detector_type = None
        for detector_type in DETECTOR_TYPES:
            key = (profile.name, detector_type)
            if key in all_experts:
                f1 = all_experts[key]['best_trial_value']
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = all_experts[key]
                    best_detector_type = detector_type
        if best_config:
            best_experts.append(best_config)
            logger.info(f"Best expert for {profile.name}: "
                        f"{best_detector_type} (F1={best_f1:.4f})")

    phase2_csv = os.path.join(args.output_dir, "phase2_cross_dd_ensemble.csv")
    if len(best_experts) > 0:
        result = evaluate_ensemble(
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=args.stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            eval_indices=eval_indices,
            expert_configs=best_experts,
            detector_seed=args.seed,
        )
        result['detector_type'] = 'mixed'
        result['ensemble_type'] = 'cross_dd_best'
        with open(phase2_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            writer.writeheader()
            writer.writerow(result)
        logger.info(f"Cross-DD best ensemble: macroF1={result['macro_f1']:.4f}")
        logger.info(f"Phase 2 results saved to {phase2_csv}")

    # ------------------------------------------------------------------
    # Phase 3: Generalist evaluation on eval set
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Phase 3: Generalist evaluation on eval set")
    logger.info("=" * 80)

    generalist_eval_csv = os.path.join(args.output_dir, "generalists_eval.csv")
    if os.path.exists(generalist_eval_csv):
        os.remove(generalist_eval_csv)

    generalist_eval_results = []
    for detector_type in DETECTOR_TYPES:
        if detector_type not in all_generalists:
            continue
        config = all_generalists[detector_type]
        result = evaluate_ensemble(
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=args.stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            eval_indices=eval_indices,
            expert_configs=[config],
            detector_seed=args.seed,
        )
        result['detector_type'] = detector_type
        result['ensemble_type'] = 'generalist'
        generalist_eval_results.append(result)
        _append_result_csv(generalist_eval_csv, result,
                           len(generalist_eval_results) == 1)
        logger.info(f"Generalist {detector_type}: macroF1={result['macro_f1']:.4f}")

    logger.info(f"Generalist eval results saved to {generalist_eval_csv}")

    logger.info("=" * 80)
    logger.info("Evaluation complete.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
