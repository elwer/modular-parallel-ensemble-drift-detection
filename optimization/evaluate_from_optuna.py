"""
Evaluate expert ensembles from an existing Optuna DB.

This script reads completed expert studies from the Optuna database,
optimizes generalist detectors per-DD (as soon as that DD's experts are
all complete), and runs ensemble evaluations incrementally.

It can be run in parallel with the expert optimization script — no need
to wait for slow DD types. Results are written to a separate output
directory.

Usage:
    python optimization/evaluate_from_optuna.py \
        --optuna-storage sqlite:///expert_ensemble_optuna.db \
        --n-streams 10 \
        --drift-frequencies 200,400,500,750,1000,1250,1500,2000,2500,3000 \
        --stream-length 8000 \
        --eval-stream-indices 1,4,8 \
        --generators SineClusters,WaveformDrift2,... \
        --seed 1337 \
        --n-trials-expert 50 \
        --per-trial-timeout 1200 \
        --output-dir expert_ensemble_eval \
        --n-jobs 7 \
        [--wait] [--poll-interval 60]
"""

import os
import sys
import csv
import json
import time
import logging
import signal as signal_module
from argparse import ArgumentParser
from typing import Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.synthetic_f1_multistream_optimize_optuna import (
    _resolve_stream_seeds,
    _default_tolerances,
    _f1_from_counts,
    evaluate_detections,
    build_stream,
    CLASS_PATH,
)
from optimization.expert_ensemble_optuna import (
    StreamProfile,
    DEFAULT_PROFILES,
    DETECTOR_TYPES,
    get_profile_indices,
    optimize_generalist_detector,
    _run_mopedds_stream,
    evaluate_ensemble,
    _append_result_csv,
    _rewrite_csv,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_expert_from_study(optuna_storage: str,
                           profile_name: str,
                           detector_type: str) -> Dict:
    """Load best trial from an existing expert Optuna study."""
    study_name = f"expert_{profile_name}_{detector_type.lower()}"
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=optuna_storage,
        )
    except KeyError:
        return None

    complete_trials = [t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) == 0:
        return None

    best_trial = study.best_trial
    return {
        'profile_name': profile_name,
        'detector_type': detector_type,
        'best_trial_value': best_trial.value,
        'best_params': dict(best_trial.params),
        'n_completed_trials': len(complete_trials),
    }


def load_generalist_from_study(optuna_storage: str,
                               detector_type: str) -> Dict:
    """Load best trial from an existing generalist Optuna study."""
    study_name = f"generalist_{detector_type.lower()}"
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=optuna_storage,
        )
    except KeyError:
        return None

    complete_trials = [t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) == 0:
        return None

    best_trial = study.best_trial
    return {
        'detector_type': detector_type,
        'best_trial_value': best_trial.value,
        'best_params': dict(best_trial.params),
        'n_completed_trials': len(complete_trials),
    }


def check_experts_complete(optuna_storage: str,
                           profiles: List[StreamProfile],
                           detector_type: str,
                           n_trials_expected: int) -> Tuple[bool, Dict]:
    """Check if all expert profiles for a detector type are complete."""
    experts = {}
    all_complete = True
    for profile in profiles:
        result = load_expert_from_study(optuna_storage, profile.name, detector_type)
        if result is None:
            all_complete = False
        else:
            key = (profile.name, detector_type)
            experts[key] = result
            if result['n_completed_trials'] < n_trials_expected:
                all_complete = False
    return all_complete, experts


def check_generalist_complete(optuna_storage: str,
                              detector_type: str,
                              n_trials_expected: int) -> bool:
    """Check if generalist study for a detector type is complete."""
    result = load_generalist_from_study(optuna_storage, detector_type)
    if result is None:
        return False
    return result['n_completed_trials'] >= n_trials_expected


def main():
    ap = ArgumentParser(description="Evaluate expert ensembles from Optuna DB")
    ap.add_argument("--optuna-storage", required=True)
    ap.add_argument("--n-streams", type=int, default=10)
    ap.add_argument("--base-stream-seed", type=int, default=42)
    ap.add_argument("--drift-frequencies", required=True)
    ap.add_argument("--stream-length", type=int, default=8000)
    ap.add_argument("--stream-seeds", type=str, default=None)
    ap.add_argument("--tolerances", type=str, default=None)
    ap.add_argument("--eval-stream-indices", required=True)
    ap.add_argument("--generators", type=str, default=None)
    ap.add_argument("--generator", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-trials-expert", type=int, default=50)
    ap.add_argument("--per-trial-timeout", type=int, default=1200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--profiles", type=str, default=None)
    ap.add_argument("--n-jobs", type=int, default=7,
                   help="Number of parallel generalist optimizations")
    ap.add_argument("--wait", action="store_true",
                   help="Wait for expert studies to complete (poll mode)")
    ap.add_argument("--poll-interval", type=int, default=60,
                   help="Seconds between polls when --wait is set")
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
    tolerances = _resolve_tolerances(args.tolerances, drift_frequencies)
    eval_indices = [int(i.strip()) for i in args.eval_stream_indices.split(',')]
    train_indices = [i for i in range(args.n_streams) if i not in eval_indices]

    if args.profiles:
        profile_data = json.loads(args.profiles)
        profiles = [StreamProfile(**p) for p in profile_data]
    else:
        profiles = DEFAULT_PROFILES

    n_profiles = len(profiles)
    n_trials_generalist = n_profiles * args.n_trials_expert

    logger.info("=" * 80)
    logger.info("Evaluate from Optuna DB")
    logger.info("=" * 80)
    logger.info(f"  Storage: {args.optuna_storage}")
    logger.info(f"  Generators: {generators}")
    logger.info(f"  Drift frequencies: {drift_frequencies}")
    logger.info(f"  Stream length: {args.stream_length}")
    logger.info(f"  Train indices: {train_indices}")
    logger.info(f"  Eval indices: {eval_indices}")
    logger.info(f"  Profiles: {[p.name for p in profiles]}")
    logger.info(f"  Expert trials expected: {args.n_trials_expert}")
    logger.info(f"  Generalist trials expected: {n_trials_generalist}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  N jobs: {args.n_jobs}")
    logger.info(f"  Wait mode: {args.wait}")

    os.makedirs(args.output_dir, exist_ok=True)

    # File paths
    expert_csv = os.path.join(args.output_dir, "experts.csv")
    generalist_csv = os.path.join(args.output_dir, "generalists.csv")
    phase1_csv = os.path.join(args.output_dir, "phase1_per_dd_ensembles.csv")
    phase2_csv = os.path.join(args.output_dir, "phase2_cross_dd_ensemble.csv")
    generalist_eval_csv = os.path.join(args.output_dir, "generalists_eval.csv")

    # Accumulators
    all_experts = {}
    all_generalists = {}
    evaluated_dd_types = set()
    generalist_trained_dd_types = set()
    generalist_evaluated_dd_types = set()

    def save_experts_csv():
        results = list(all_experts.values())
        _rewrite_csv(expert_csv, results)

    def save_generalists_csv():
        results = list(all_generalists.values())
        _rewrite_csv(generalist_csv, results)

    def append_phase1(result):
        row = dict(result)
        file_exists = os.path.exists(phase1_csv)
        with open(phase1_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def append_generalist_eval(result):
        row = dict(result)
        file_exists = os.path.exists(generalist_eval_csv)
        with open(generalist_eval_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def evaluate_per_dd(detector_type, experts_for_dd):
        """Evaluate per-DD ensemble for a single detector type."""
        expert_configs = list(experts_for_dd.values())
        if len(expert_configs) == 0:
            return

        logger.info(f"Evaluating per-DD ensemble for {detector_type} "
                    f"({len(expert_configs)} experts)")
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
        append_phase1(result)
        logger.info(f"Per-DD ensemble {detector_type}: "
                     f"macroF1={result['macro_f1']:.4f} (saved)")

    def evaluate_generalist_single(detector_type):
        """Evaluate a single generalist detector on eval set."""
        config = all_generalists[detector_type]
        logger.info(f"Evaluating generalist {detector_type} on eval set")
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
        append_generalist_eval(result)
        logger.info(f"Generalist {detector_type}: "
                     f"macroF1={result['macro_f1']:.4f} (saved)")

    def evaluate_cross_dd():
        """Evaluate cross-DD best ensemble."""
        if len(all_experts) == 0:
            return

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

        if len(best_experts) == 0:
            return

        logger.info(f"Evaluating cross-DD best ensemble ({len(best_experts)} experts)")
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
        logger.info(f"Cross-DD best ensemble: macroF1={result['macro_f1']:.4f} (saved)")

    # Main loop
    while True:
        # Phase 1: Load completed experts from Optuna DB
        new_experts_found = False
        for detector_type in DETECTOR_TYPES:
            for profile in profiles:
                key = (profile.name, detector_type)
                if key in all_experts:
                    continue
                result = load_expert_from_study(
                    args.optuna_storage, profile.name, detector_type)
                if result is not None:
                    all_experts[key] = result
                    new_experts_found = True
                    logger.info(f"Loaded expert {profile.name}_{detector_type}: "
                                f"F1={result['best_trial_value']:.4f} "
                                f"({result['n_completed_trials']} trials)")

        if new_experts_found:
            save_experts_csv()
            logger.info(f"Experts CSV updated ({len(all_experts)} experts)")

        # Phase 2: For each DD type with all experts complete,
        #           start generalist optimization if not done
        generalist_tasks_to_run = []
        for detector_type in DETECTOR_TYPES:
            if detector_type in generalist_trained_dd_types:
                continue
            if detector_type in all_generalists:
                continue

            all_complete, _ = check_experts_complete(
                args.optuna_storage, profiles, detector_type, args.n_trials_expert)

            if all_complete:
                # Check if generalist study already exists and is complete
                if check_generalist_complete(
                        args.optuna_storage, detector_type, n_trials_generalist):
                    # Load from DB
                    result = load_generalist_from_study(
                        args.optuna_storage, detector_type)
                    all_generalists[detector_type] = result
                    generalist_trained_dd_types.add(detector_type)
                    save_generalists_csv()
                    logger.info(f"Generalist {detector_type} already complete "
                                f"({result['n_completed_trials']} trials), loaded")
                else:
                    generalist_tasks_to_run.append(detector_type)

        if generalist_tasks_to_run:
            logger.info(f"Starting generalist optimization for: "
                        f"{generalist_tasks_to_run}")
            tasks = []
            for detector_type in generalist_tasks_to_run:
                tasks.append({
                    'optuna_storage': args.optuna_storage,
                    'generators': generators,
                    'drift_frequencies': drift_frequencies,
                    'stream_length': args.stream_length,
                    'stream_seeds': stream_seeds,
                    'tolerances': tolerances,
                    'train_indices': train_indices,
                    'detector_type': detector_type,
                    'detector_seed': args.seed,
                    'n_trials': n_trials_generalist,
                    'per_trial_timeout': args.per_trial_timeout,
                })

            n_workers = min(args.n_jobs, len(tasks))
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(optimize_generalist_detector, **task): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    detector_type = task['detector_type']
                    result = future.result()
                    if 'error' not in result:
                        all_generalists[detector_type] = result
                        generalist_trained_dd_types.add(detector_type)
                        save_generalists_csv()
                        logger.info(f"Generalist {detector_type}: "
                                    f"F1={result['best_trial_value']:.4f} (saved)")
                    else:
                        logger.error(f"Generalist {detector_type} failed: {result.get('error')}")

        # Phase 3: Evaluate per-DD ensembles for newly complete DD types
        for detector_type in DETECTOR_TYPES:
            if detector_type in evaluated_dd_types:
                continue

            # Check if all expert profiles for this DD are loaded
            experts_for_dd = {}
            for profile in profiles:
                key = (profile.name, detector_type)
                if key in all_experts:
                    experts_for_dd[key] = all_experts[key]

            if len(experts_for_dd) == len(profiles):
                evaluate_per_dd(detector_type, experts_for_dd)
                evaluated_dd_types.add(detector_type)

        # Phase 4: Evaluate generalists on eval set
        for detector_type in DETECTOR_TYPES:
            if detector_type in generalist_evaluated_dd_types:
                continue
            if detector_type not in all_generalists:
                continue

            evaluate_generalist_single(detector_type)
            generalist_evaluated_dd_types.add(detector_type)

        # Phase 5: Cross-DD evaluation (once all DD types are evaluated)
        if (len(evaluated_dd_types) == len(DETECTOR_TYPES)
                and not os.path.exists(phase2_csv)):
            evaluate_cross_dd()

        # Check if everything is done
        all_done = (
            len(evaluated_dd_types) == len(DETECTOR_TYPES)
            and len(generalist_evaluated_dd_types) == len(DETECTOR_TYPES)
            and os.path.exists(phase2_csv)
        )

        if all_done:
            logger.info("=" * 80)
            logger.info("All evaluations complete!")
            logger.info("=" * 80)
            break

        if not args.wait:
            logger.info("=" * 80)
            logger.info(f"Snapshot complete. "
                        f"Experts: {len(all_experts)}, "
                        f"Generalists: {len(all_generalists)}, "
                        f"Per-DD evaluated: {len(evaluated_dd_types)}/{len(DETECTOR_TYPES)}, "
                        f"Generalist evaluated: {len(generalist_evaluated_dd_types)}/{len(DETECTOR_TYPES)}, "
                        f"Cross-DD: {'done' if os.path.exists(phase2_csv) else 'pending'}")
            logger.info("Use --wait to keep polling until all studies complete.")
            logger.info("=" * 80)
            break

        logger.info(f"Waiting {args.poll_interval}s before next poll... "
                     f"(experts: {len(all_experts)}/{len(profiles)*len(DETECTOR_TYPES)}, "
                     f"generalists: {len(all_generalists)}/{len(DETECTOR_TYPES)})")
        time.sleep(args.poll_interval)


def _resolve_tolerances(tolerances_str, drift_frequencies):
    if tolerances_str:
        return [int(t.strip()) for t in tolerances_str.split(',')]
    return _default_tolerances(drift_frequencies)


if __name__ == "__main__":
    main()
