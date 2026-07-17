"""
Compare results across all DD types and build the cross-DD best ensemble.

Loads expert and generalist results from the Optuna DB and per-DD CSV files, then:
  1. Prints a comparison table of per-DD ensembles vs generalists
     (Phase 1b fixed params, Phase 2 optimized params, generalist single+optimized)
  2. Builds the cross-DD best ensemble (best expert per profile across all DD types)
  3. Evaluates it on the eval set with fixed and optimized deployment params
  4. Saves everything to CSV

Run this after all expert and generalist jobs have completed.

Usage:
    python split_pipeline/compare_results.py \
        --optuna-storage sqlite:///split_pipeline/split_pipeline_optuna.db \
        --n-streams 10 --base-stream-seed 42 \
        --drift-frequencies 200,400,500,750,1000,1250,1500,2000,2500,3000 \
        --stream-length 8000 --eval-stream-indices 1,4,8 \
        --generators SineClusters,WaveformDrift2,... \
        --output-dir split_pipeline/results
"""

import os
import sys
import csv
import json
import logging
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
    evaluate_ensemble,
    _append_result_csv,
)
from optimization.evaluate_from_optuna import (
    load_expert_from_study,
    load_generalist_from_study,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = ArgumentParser(description="Compare results and build cross-DD best ensemble")
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
    stream_seeds = _resolve_stream_seeds(args.stream_seeds, args.n_streams,
                                         args.base_stream_seed)
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
    logger.info("Cross-DD comparison and best-of-all ensemble")
    logger.info("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all experts from DB
    # ------------------------------------------------------------------
    all_experts = {}
    for detector_type in DETECTOR_TYPES:
        for profile in profiles:
            key = (profile.name, detector_type)
            result = load_expert_from_study(
                args.optuna_storage, profile.name, detector_type)
            if result is not None:
                all_experts[key] = result

    logger.info(f"Loaded {len(all_experts)} experts from DB")

    # ------------------------------------------------------------------
    # Load all generalists from DB
    # ------------------------------------------------------------------
    all_generalists = {}
    for detector_type in DETECTOR_TYPES:
        result = load_generalist_from_study(
            args.optuna_storage, detector_type)
        if result is not None:
            all_generalists[detector_type] = result

    logger.info(f"Loaded {len(all_generalists)} generalists from DB")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'DD Type':<12} {'Expert Train F1':>16} {'Generalist Train F1':>20}")
    print("-" * 50)

    for detector_type in DETECTOR_TYPES:
        # Best expert train F1 for this DD type
        expert_f1s = []
        for profile in profiles:
            key = (profile.name, detector_type)
            if key in all_experts:
                expert_f1s.append(all_experts[key]['best_trial_value'])
        expert_str = f"{sum(expert_f1s)/len(expert_f1s):.4f}" if expert_f1s else "N/A"

        # Generalist train F1
        gen_str = f"{all_generalists[detector_type]['best_trial_value']:.4f}" \
            if detector_type in all_generalists else "N/A"

        print(f"{detector_type:<12} {expert_str:>16} {gen_str:>20}")

    # Load per-DD Phase 1b (fixed params) eval results from CSV
    per_dd_fixed = {}
    for dd in DETECTOR_TYPES:
        csv_path = os.path.join(args.output_dir, f"phase1_per_dd_{dd.lower()}.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                row = csv.DictReader(f).__next__()
                per_dd_fixed[dd] = float(row['macro_f1'])

    # Load per-DD Phase 2 (optimized params) eval results from CSV
    per_dd_opt = {}
    for dd in DETECTOR_TYPES:
        csv_path = os.path.join(args.output_dir, f"phase2_per_dd_{dd.lower()}.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                row = csv.DictReader(f).__next__()
                per_dd_opt[dd] = float(row['macro_f1'])

    # Load generalist single-detector eval results from CSV
    gen_single = {}
    for dd in DETECTOR_TYPES:
        csv_path = os.path.join(args.output_dir, f"generalists_eval_{dd.lower()}.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                row = csv.DictReader(f).__next__()
                gen_single[dd] = float(row['macro_f1'])

    print(f"\n{'DD Type':<12} {'Ens Fixed':>10} {'Ens Opt':>10} {'Gen Single':>12}")
    print("-" * 46)
    for dd in DETECTOR_TYPES:
        ef = f"{per_dd_fixed[dd]:.4f}" if dd in per_dd_fixed else "N/A"
        eo = f"{per_dd_opt[dd]:.4f}" if dd in per_dd_opt else "N/A"
        gs = f"{gen_single[dd]:.4f}" if dd in gen_single else "N/A"
        print(f"{dd:<12} {ef:>10} {eo:>10} {gs:>12}")

    # ------------------------------------------------------------------
    # Build cross-DD best ensemble (best expert per profile across all DD types)
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Building cross-DD best ensemble")
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
                        f"{best_detector_type} (train F1={best_f1:.4f})")

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

        phase2_csv = os.path.join(args.output_dir, "cross_dd_best_ensemble.csv")
        with open(phase2_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            writer.writeheader()
            writer.writerow(result)
        logger.info(f"Cross-DD best ensemble: macroF1={result['macro_f1']:.4f}")
        logger.info(f"Saved to {phase2_csv}")

        print(f"\n{'Cross-DD':<12} {result['macro_f1']:>10.4f} {'(fixed)':>10}")
    else:
        logger.warning("No experts available for cross-DD ensemble")

    print("\n" + "=" * 80)
    print("Comparison complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
