"""
Expert optimization + per-DD ensemble evaluation for a single detector type.

Phase 1: Optimize K profile-experts for one DD type (7 studies × 100 trials each).
Phase 1b: Evaluate per-DD ensemble with fixed conservative deployment params.
Phase 2: Optuna optimization of ensemble deployment params (100 trials):
         detector_decision_criteria, ensemble_decision_criteria,
         decision_window, suppression_window, recent_samples_size.
Designed to be submitted as one SLURM job per detector type (7 jobs total).

Usage:
    python split_pipeline/expert_optimize.py \
        --optuna-storage sqlite:///split_pipeline/split_pipeline_optuna.db \
        --detector-type BNDM \
        --n-streams 10 --base-stream-seed 42 \
        --drift-frequencies 200,400,500,750,1000,1250,1500,2000,2500,3000 \
        --stream-length 8000 --eval-stream-indices 1,4,8 \
        --generators SineClusters,WaveformDrift2,... \
        --n-trials-expert 100 --n-trials-deployment 100 \
        --per-trial-timeout 1200 \
        --n-jobs 7 --output-dir split_pipeline/results
"""

import os
import sys
import csv
import json
import logging
import optuna
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
    _run_mopedds_stream,
    _f1_from_counts,
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
    ap.add_argument("--n-trials-expert", type=int, default=100)
    ap.add_argument("--n-trials-deployment", type=int, default=100,
                   help="Phase 2: trials for ensemble deployment param optimization")
    ap.add_argument("--per-trial-timeout", type=int, default=1200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--profiles", type=str, default=None,
                   help="JSON string defining custom profiles")
    ap.add_argument("--n-jobs", type=int, default=7,
                   help="Number of parallel profile optimizations (default: 7 = one per profile)")
    ap.add_argument("--load-if-exists", action="store_true",
                   help="Resume existing Optuna studies instead of creating fresh ones")
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
    expert_csv = os.path.join(args.output_dir, f"experts_{args.detector_type.lower()}.csv")

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
            'load_if_exists': args.load_if_exists,
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
    # Phase 1b: Evaluate per-DD ensemble with fixed conservative params
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"Phase 1b: Per-DD ensemble eval (fixed params) for {args.detector_type}")
    logger.info("=" * 80)

    expert_configs = [experts[name] for name in sorted(experts)]
    if len(expert_configs) == 0:
        logger.warning("No experts available, skipping ensemble evaluation")
        return

    slot_specs = []
    for config in expert_configs:
        slot_specs.append((config['detector_type'], config['best_params']))

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
    ens_result['ensemble_type'] = 'per_dd_fixed'

    phase1_csv = os.path.join(args.output_dir, f"phase1_per_dd_{args.detector_type.lower()}.csv")
    _append_result_csv(phase1_csv, ens_result, True)
    logger.info(f"Phase 1b ensemble {args.detector_type}: "
                f"macroF1={ens_result['macro_f1']:.4f} (saved to {phase1_csv})")

    # ------------------------------------------------------------------
    # Phase 2: Optuna optimization of ensemble deployment params
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"Phase 2: Optimizing deployment params for {args.detector_type} ensemble")
    logger.info("=" * 80)

    train_indices = [i for i in range(args.n_streams) if i not in eval_set]

    def deployment_objective(trial):
        det_criteria = trial.suggest_categorical("detector_decision_criteria",
                                                  ["any", "all", "majority"])
        ens_criteria = trial.suggest_categorical("ensemble_decision_criteria",
                                                  ["any", "all", "majority"])
        decision_window = trial.suggest_int("decision_window", 1, 50)
        suppression_window = trial.suggest_int("suppression_window", 0, 100)
        recent_samples_size = trial.suggest_int("recent_samples_size", 50, 500)

        per_stream_f1 = []
        for s_idx in train_indices:
            tp, fp, fn, _, f1, _, _, _ = _run_mopedds_stream(
                generator_name=generators[s_idx],
                drift_frequency=drift_frequencies[s_idx],
                stream_length=args.stream_length,
                stream_seed=stream_seeds[s_idx],
                tolerance=tolerances[s_idx],
                slot_specs=slot_specs,
                detector_seed_base=args.seed,
                s_idx=s_idx,
                detector_decision_criteria=det_criteria,
                ensemble_decision_criteria=ens_criteria,
                decision_window=decision_window,
                suppression_window=suppression_window,
                recent_samples_size=recent_samples_size,
            )
            per_stream_f1.append(f1)
        return sum(per_stream_f1) / len(per_stream_f1) if per_stream_f1 else 0.0

    study_name = f"deploy_expert_{args.detector_type.lower()}"
    dep_study = optuna.create_study(
        study_name=study_name,
        storage=args.optuna_storage,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        direction="maximize",
        load_if_exists=args.load_if_exists,
    )
    completed_deployment_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in dep_study.trials
    )
    remaining_deployment_trials = max(
        0, args.n_trials_deployment - completed_deployment_trials)
    if remaining_deployment_trials:
        dep_study.optimize(deployment_objective, n_trials=remaining_deployment_trials,
                           n_jobs=1, show_progress_bar=True)
    else:
        logger.info("Deployment study already has the requested number of trials")

    best_dep = dep_study.best_trial
    logger.info(f"Phase 2 best deployment params: {best_dep.params}")
    logger.info(f"Phase 2 best train F1: {best_dep.value:.4f}")

    # Evaluate best deployment params on eval set
    tp_total, fp_total, fn_total = 0, 0, 0
    per_stream_f1 = []
    for s_idx in eval_indices:
        tp, fp, fn, _, f1, _, _, _ = _run_mopedds_stream(
            generator_name=generators[s_idx],
            drift_frequency=drift_frequencies[s_idx],
            stream_length=args.stream_length,
            stream_seed=stream_seeds[s_idx],
            tolerance=tolerances[s_idx],
            slot_specs=slot_specs,
            detector_seed_base=args.seed,
            s_idx=s_idx,
            detector_decision_criteria=best_dep.params["detector_decision_criteria"],
            ensemble_decision_criteria=best_dep.params["ensemble_decision_criteria"],
            decision_window=best_dep.params["decision_window"],
            suppression_window=best_dep.params["suppression_window"],
            recent_samples_size=best_dep.params["recent_samples_size"],
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
        per_stream_f1.append(f1)

    macro_f1 = sum(per_stream_f1) / len(per_stream_f1) if per_stream_f1 else 0.0
    micro_f1 = _f1_from_counts(tp_total, fp_total, fn_total)

    phase2_result = {
        'detector_type': args.detector_type,
        'ensemble_type': 'per_dd_optimized',
        'train_f1': best_dep.value,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'tp_total': tp_total,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'per_stream_f1': per_stream_f1,
        'best_deployment_params': json.dumps(best_dep.params),
    }
    phase2_csv = os.path.join(args.output_dir, f"phase2_per_dd_{args.detector_type.lower()}.csv")
    _append_result_csv(phase2_csv, phase2_result, True)
    logger.info(f"Phase 2 ensemble {args.detector_type}: "
                f"macroF1={macro_f1:.4f} (saved to {phase2_csv})")

    logger.info("Expert optimization + deployment optimization complete.")


if __name__ == "__main__":
    main()
