"""
Expert Ensemble Optimization using Optuna.

This script implements the expert ensemble approach:
1. Define K stream profiles based on characteristics (drift frequency ranges, generator types)
2. Train K × 7 experts (single detectors) - 50 trials each, 20min timeout per trial
3. Train 7 generalist single DDs - K × 50 trials each (same total budget)
4. Evaluate using MoPEDDs ensemble with "any" criterion

Evaluation phases:
- Phase 1: Per-DD ensembles (K experts of same DD type)
- Phase 2: Cross-DD best experts (best expert per profile across all DD types)
"""

import os
import sys
import csv
import logging
import json
import signal
from argparse import ArgumentParser
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import optuna
from optuna.samplers import TPESampler

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.synthetic_f1_multistream_optimize_optuna import (
    _resolve_stream_seeds,
    _resolve_list,
    _resolve_generators,
    _resolve_study_tag,
    _default_tolerances,
    GENERATORS,
    _suggest_detector_params,
    _instantiate,
    _f1_from_counts,
    _run_one_stream,
    apply_suppression,
    evaluate_detections,
    run_ensemble,
)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Detector types
DETECTOR_TYPES = ["BNDM", "CSDDM", "D3", "IBDD", "OCDD", "SPLL", "UDetect"]


@dataclass
class StreamProfile:
    """Defines a stream profile for expert specialization."""
    name: str
    generator_filter: str  # "SineClusters", "WaveformDrift2", or "all"
    drift_freq_min: int
    drift_freq_max: int


# Default profiles: one per train stream (unique generator + drift frequency)
# Train streams: 0(SineClusters,200), 2(SineClusters,500), 3(WaveformDrift2,750),
#                5(WaveformDrift2,1250), 6(SineClusters,1500), 7(WaveformDrift2,2000),
#                9(WaveformDrift2,3000)
# Eval streams 1,4,8 must NOT match any profile.
DEFAULT_PROFILES = [
    StreamProfile("sineclusters_200", "SineClusters", 200, 200),
    StreamProfile("sineclusters_500", "SineClusters", 500, 500),
    StreamProfile("waveform_750", "WaveformDrift2", 750, 750),
    StreamProfile("waveform_1250", "WaveformDrift2", 1250, 1250),
    StreamProfile("sineclusters_1500", "SineClusters", 1500, 1500),
    StreamProfile("waveform_2000", "WaveformDrift2", 2000, 2000),
    StreamProfile("waveform_3000", "WaveformDrift2", 3000, 3000),
]


class _TrialTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TrialTimeout("Trial exceeded per-trial time limit")


def get_profile_indices(profiles: List[StreamProfile],
                        generators: List[str],
                        drift_frequencies: List[int]) -> Dict[str, List[int]]:
    """Get stream indices for each profile."""
    profile_indices = {profile.name: [] for profile in profiles}
    
    for i, (gen, freq) in enumerate(zip(generators, drift_frequencies)):
        for profile in profiles:
            gen_match = (profile.generator_filter == "all" or 
                        profile.generator_filter == gen)
            freq_match = profile.drift_freq_min <= freq <= profile.drift_freq_max
            if gen_match and freq_match:
                profile_indices[profile.name].append(i)
                break  # Each stream belongs to first matching profile
    
    return profile_indices


def optimize_single_detector_expert(*,
                                   optuna_storage: str,
                                   generators: List[str],
                                   drift_frequencies: List[int],
                                   stream_length: int,
                                   stream_seeds: List[int],
                                   tolerances: List[int],
                                   profile_name: str,
                                   profile_indices: List[int],
                                   detector_type: str,
                                   detector_seed: int,
                                   n_trials: int = 50,
                                   per_trial_timeout: int = 1200) -> Dict:
    """Optimize a single detector expert for a specific profile."""
    logger.info(f"Optimizing {detector_type} expert for profile {profile_name}")
    logger.info(f"  Profile indices: {profile_indices}")
    logger.info(f"  Trials: {n_trials}, Timeout: {per_trial_timeout}s")
    
    if len(profile_indices) == 0:
        logger.warning(f"No streams match profile {profile_name}, skipping")
        return {'error': 'no_streams', 'profile_name': profile_name, 'detector_type': detector_type}
    
    # Create Optuna study
    import uuid
    run_id = str(uuid.uuid4())[:8]
    study_name = f"expert_{profile_name}_{detector_type.lower()}_{run_id}"
    study = optuna.create_study(
        study_name=study_name,
        storage=optuna_storage,
        sampler=TPESampler(seed=detector_seed),
        direction="maximize",
        load_if_exists=False,
    )
    
    def objective(trial: optuna.Trial) -> float:
        # Sample detector hyperparameters
        params = _suggest_detector_params(trial, "", detector_type)
        
        # Fixed conservative MoPEDDS parameters (not optimized)
        detector_decision_criteria = "any"
        ensemble_decision_criteria = "any"
        decision_window = 1
        suppression_window = 0
        recent_samples_size = 100
        
        # Build slot specs for single detector
        slot_specs = [(detector_type, params)]
        
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(per_trial_timeout)
        
        try:
            tp_total = 0
            fp_total = 0
            fn_total = 0
            per_stream_f1 = []
            
            for s_idx in profile_indices:
                tp, fp, fn, mean_delay, f1, prec, rec, n_known = _run_one_stream(
                    generator_name=generators[s_idx],
                    drift_frequency=drift_frequencies[s_idx],
                    stream_length=stream_length,
                    stream_seed=stream_seeds[s_idx],
                    tolerance=tolerances[s_idx],
                    slot_specs=slot_specs,
                    detector_seed_base=detector_seed,
                    s_idx=s_idx,
                    detector_criterion=detector_decision_criteria,
                    ensemble_criterion=ensemble_decision_criteria,
                    decision_window=decision_window,
                    suppression_window=suppression_window,
                    recent_samples_size=recent_samples_size,
                )
                tp_total += tp
                fp_total += fp
                fn_total += fn
                per_stream_f1.append(f1)
            
            macro_f1 = sum(per_stream_f1) / len(per_stream_f1) if per_stream_f1 else 0.0
            logger.info(f"Trial {trial.number}: macroF1={macro_f1:.4f} on profile {profile_name}")
            
            trial.set_user_attr("macro_f1", macro_f1)
            trial.set_user_attr("profile_name", profile_name)
            trial.set_user_attr("detector_type", detector_type)
            
            return macro_f1
            
        except _TrialTimeout:
            logger.warning(f"Trial {trial.number} timed out")
            trial.set_user_attr("error", "timeout")
            return 0.0
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e!r}")
            trial.set_user_attr("error", repr(e))
            return 0.0
        finally:
            signal.alarm(0)
    
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    except Exception as e:
        logger.warning(f"Optuna optimization failed: {e}")
        return {'error': str(e), 'profile_name': profile_name, 'detector_type': detector_type}
    
    best_trial = study.best_trial
    logger.info(f"Best trial F1 = {best_trial.value:.4f}")
    
    # All trial params are detector hyperparameters (MoPEDDS params are fixed)
    best_params = dict(best_trial.params)
    
    return {
        'profile_name': profile_name,
        'detector_type': detector_type,
        'best_trial_value': best_trial.value,
        'best_params': best_params,
    }


def optimize_generalist_detector(*,
                                 optuna_storage: str,
                                 generators: List[str],
                                 drift_frequencies: List[int],
                                 stream_length: int,
                                 stream_seeds: List[int],
                                 tolerances: List[int],
                                 train_indices: List[int],
                                 detector_type: str,
                                 detector_seed: int,
                                 n_trials: int,
                                 per_trial_timeout: int = 1200) -> Dict:
    """Optimize a generalist detector on all streams."""
    logger.info(f"Optimizing generalist {detector_type}")
    logger.info(f"  Train indices: {train_indices}")
    logger.info(f"  Trials: {n_trials}, Timeout: {per_trial_timeout}s")
    
    # Create Optuna study
    import uuid
    run_id = str(uuid.uuid4())[:8]
    study_name = f"generalist_{detector_type.lower()}_{run_id}"
    study = optuna.create_study(
        study_name=study_name,
        storage=optuna_storage,
        sampler=TPESampler(seed=detector_seed),
        direction="maximize",
        load_if_exists=False,
    )
    
    def objective(trial: optuna.Trial) -> float:
        # Sample detector hyperparameters
        params = _suggest_detector_params(trial, "", detector_type)
        
        # Fixed conservative MoPEDDS parameters (not optimized)
        detector_decision_criteria = "any"
        ensemble_decision_criteria = "any"
        decision_window = 1
        suppression_window = 0
        recent_samples_size = 100
        
        # Build slot specs for single detector
        slot_specs = [(detector_type, params)]
        
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(per_trial_timeout)
        
        try:
            tp_total = 0
            fp_total = 0
            fn_total = 0
            per_stream_f1 = []
            
            for s_idx in train_indices:
                tp, fp, fn, mean_delay, f1, prec, rec, n_known = _run_one_stream(
                    generator_name=generators[s_idx],
                    drift_frequency=drift_frequencies[s_idx],
                    stream_length=stream_length,
                    stream_seed=stream_seeds[s_idx],
                    tolerance=tolerances[s_idx],
                    slot_specs=slot_specs,
                    detector_seed_base=detector_seed,
                    s_idx=s_idx,
                    detector_criterion=detector_decision_criteria,
                    ensemble_criterion=ensemble_decision_criteria,
                    decision_window=decision_window,
                    suppression_window=suppression_window,
                    recent_samples_size=recent_samples_size,
                )
                tp_total += tp
                fp_total += fp
                fn_total += fn
                per_stream_f1.append(f1)
            
            macro_f1 = sum(per_stream_f1) / len(per_stream_f1) if per_stream_f1 else 0.0
            logger.info(f"Trial {trial.number}: macroF1={macro_f1:.4f} (generalist)")
            
            trial.set_user_attr("macro_f1", macro_f1)
            trial.set_user_attr("detector_type", detector_type)
            
            return macro_f1
            
        except _TrialTimeout:
            logger.warning(f"Trial {trial.number} timed out")
            trial.set_user_attr("error", "timeout")
            return 0.0
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e!r}")
            trial.set_user_attr("error", repr(e))
            return 0.0
        finally:
            signal.alarm(0)
    
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    except Exception as e:
        logger.warning(f"Optuna optimization failed: {e}")
        return {'error': str(e), 'detector_type': detector_type}
    
    best_trial = study.best_trial
    logger.info(f"Best trial F1 = {best_trial.value:.4f}")
    
    # All trial params are detector hyperparameters (MoPEDDS params are fixed)
    best_params = dict(best_trial.params)
    
    return {
        'detector_type': detector_type,
        'best_trial_value': best_trial.value,
        'best_params': best_params,
    }


def evaluate_ensemble(*,
                      generators: List[str],
                      drift_frequencies: List[int],
                      stream_length: int,
                      stream_seeds: List[int],
                      tolerances: List[int],
                      eval_indices: List[int],
                      expert_configs: List[Dict],
                      detector_seed: int) -> Dict:
    """Evaluate MoPEDDs ensemble with experts using 'any' criterion."""
    logger.info(f"Evaluating ensemble with {len(expert_configs)} experts")
    
    if len(expert_configs) == 0:
        return {'error': 'no_experts'}
    
    # Build slot specs from expert configs
    slot_specs = []
    for config in expert_configs:
        detector_type = config['detector_type']
        params = config['best_params']
        slot_specs.append((detector_type, params))
    
    # Use fixed ensemble parameters
    detector_criterion = "any"
    ensemble_criterion = "any"
    decision_window = 1
    suppression_window = 0
    recent_samples_size = 100
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    per_stream_f1 = []
    per_stream_tp = []
    per_stream_fp = []
    per_stream_fn = []
    
    for s_idx in eval_indices:
        tp, fp, fn, mean_delay, f1, prec, rec, n_known = _run_one_stream(
            generator_name=generators[s_idx],
            drift_frequency=drift_frequencies[s_idx],
            stream_length=stream_length,
            stream_seed=stream_seeds[s_idx],
            tolerance=tolerances[s_idx],
            slot_specs=slot_specs,
            detector_seed_base=detector_seed,
            s_idx=s_idx,
            detector_criterion=detector_criterion,
            ensemble_criterion=ensemble_criterion,
            decision_window=decision_window,
            suppression_window=suppression_window,
            recent_samples_size=recent_samples_size,
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
        per_stream_f1.append(f1)
        per_stream_tp.append(tp)
        per_stream_fp.append(fp)
        per_stream_fn.append(fn)
    
    macro_f1 = sum(per_stream_f1) / len(per_stream_f1) if per_stream_f1 else 0.0
    micro_f1 = _f1_from_counts(tp_total, fp_total, fn_total)
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'tp_total': tp_total,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'per_stream_f1': per_stream_f1,
        'per_stream_tp': per_stream_tp,
        'per_stream_fp': per_stream_fp,
        'per_stream_fn': per_stream_fn,
    }


def main():
    ap = ArgumentParser(description="Expert Ensemble Optimization")
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
    ap.add_argument("--per-trial-timeout", type=int, default=1200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--profiles", type=str, default=None,
                   help="JSON string defining custom profiles")
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
    
    # Parse drift frequencies
    drift_frequencies = [int(f.strip()) for f in args.drift_frequencies.split(',')]
    
    # Parse stream seeds
    if args.stream_seeds:
        stream_seeds = [int(s.strip()) for s in args.stream_seeds.split(',')]
    else:
        stream_seeds = [args.base_stream_seed + i for i in range(args.n_streams)]
    
    # Parse tolerances
    if args.tolerances:
        tolerances = [int(t.strip()) for t in args.tolerances.split(',')]
    else:
        tolerances = _default_tolerances(drift_frequencies)
    
    # Parse eval indices
    eval_indices = [int(i.strip()) for i in args.eval_stream_indices.split(',')]
    train_indices = [i for i in range(args.n_streams) if i not in eval_indices]
    
    # Parse profiles
    if args.profiles:
        profile_data = json.loads(args.profiles)
        profiles = [StreamProfile(**p) for p in profile_data]
    else:
        profiles = DEFAULT_PROFILES
    
    logger.info(f"Configuration:")
    logger.info(f"  Generators: {generators}")
    logger.info(f"  N streams: {args.n_streams}")
    logger.info(f"  Drift frequencies: {drift_frequencies}")
    logger.info(f"  Stream length: {args.stream_length}")
    logger.info(f"  Train indices: {train_indices}")
    logger.info(f"  Eval indices: {eval_indices}")
    logger.info(f"  Profiles: {[p.name for p in profiles]}")
    logger.info(f"  Expert trials: {args.n_trials_expert}")
    logger.info(f"  Per-trial timeout: {args.per_trial_timeout}s")
    
    # Get profile indices (filter out eval streams to prevent data leakage)
    eval_set = set(eval_indices)
    profile_indices = get_profile_indices(profiles, generators, drift_frequencies)
    for profile_name in profile_indices:
        profile_indices[profile_name] = [i for i in profile_indices[profile_name] if i not in eval_set]
    for profile_name, indices in profile_indices.items():
        logger.info(f"  Profile {profile_name}: {len(indices)} streams (indices: {indices})")
    
    # Calculate generalist trials (K * 50)
    n_profiles = len(profiles)
    n_trials_generalist = n_profiles * args.n_trials_expert
    logger.info(f"  Generalist trials: {n_trials_generalist}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 2: Train K × 7 experts
    logger.info("=" * 80)
    logger.info("STEP 2: Training experts")
    logger.info("=" * 80)
    
    experts = {}  # (profile_name, detector_type) -> config
    expert_results = []
    
    for profile in profiles:
        for detector_type in DETECTOR_TYPES:
            result = optimize_single_detector_expert(
                optuna_storage=args.optuna_storage,
                generators=generators,
                drift_frequencies=drift_frequencies,
                stream_length=args.stream_length,
                stream_seeds=stream_seeds,
                tolerances=tolerances,
                profile_name=profile.name,
                profile_indices=profile_indices[profile.name],
                detector_type=detector_type,
                detector_seed=args.seed,
                n_trials=args.n_trials_expert,
                per_trial_timeout=args.per_trial_timeout,
            )
            
            if 'error' not in result:
                experts[(profile.name, detector_type)] = result
                expert_results.append(result)
                logger.info(f"Expert {profile.name}_{detector_type}: F1={result['best_trial_value']:.4f}")
    
    # Save expert results
    expert_csv = os.path.join(args.output_dir, "experts.csv")
    with open(expert_csv, 'w', newline='') as f:
        if expert_results:
            writer = csv.DictWriter(f, fieldnames=expert_results[0].keys())
            writer.writeheader()
            for result in expert_results:
                writer.writerow(result)
    logger.info(f"Expert results saved to {expert_csv}")
    
    # Step 3: Train 7 generalist detectors
    logger.info("=" * 80)
    logger.info("STEP 3: Training generalist detectors")
    logger.info("=" * 80)
    
    generalists = {}  # detector_type -> config
    generalist_results = []
    
    for detector_type in DETECTOR_TYPES:
        result = optimize_generalist_detector(
            optuna_storage=args.optuna_storage,
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=args.stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            train_indices=train_indices,
            detector_type=detector_type,
            detector_seed=args.seed,
            n_trials=n_trials_generalist,
            per_trial_timeout=args.per_trial_timeout,
        )
        
        if 'error' not in result:
            generalists[detector_type] = result
            generalist_results.append(result)
            logger.info(f"Generalist {detector_type}: F1={result['best_trial_value']:.4f}")
    
    # Save generalist results
    generalist_csv = os.path.join(args.output_dir, "generalists.csv")
    with open(generalist_csv, 'w', newline='') as f:
        if generalist_results:
            writer = csv.DictWriter(f, fieldnames=generalist_results[0].keys())
            writer.writeheader()
            for result in generalist_results:
                writer.writerow(result)
    logger.info(f"Generalist results saved to {generalist_csv}")
    
    # Step 4: Evaluation Phase 1 - Per-DD ensembles
    logger.info("=" * 80)
    logger.info("STEP 4: Evaluation Phase 1 - Per-DD ensembles")
    logger.info("=" * 80)
    
    phase1_results = []
    for detector_type in DETECTOR_TYPES:
        # Build ensemble with one expert per profile for this detector type
        expert_configs = []
        for profile in profiles:
            key = (profile.name, detector_type)
            if key in experts:
                expert_configs.append(experts[key])
        
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
        logger.info(f"Per-DD ensemble {detector_type}: macroF1={result['macro_f1']:.4f}")
    
    # Save phase 1 results
    phase1_csv = os.path.join(args.output_dir, "phase1_per_dd_ensembles.csv")
    with open(phase1_csv, 'w', newline='') as f:
        if phase1_results:
            writer = csv.DictWriter(f, fieldnames=phase1_results[0].keys())
            writer.writeheader()
            for result in phase1_results:
                writer.writerow(result)
    logger.info(f"Phase 1 results saved to {phase1_csv}")
    
    # Step 4: Evaluation Phase 2 - Cross-DD best experts
    logger.info("=" * 80)
    logger.info("STEP 4: Evaluation Phase 2 - Cross-DD best experts")
    logger.info("=" * 80)
    
    # Select best expert per profile across all DD types
    best_experts = []
    for profile in profiles:
        best_f1 = -1
        best_config = None
        best_detector_type = None
        
        for detector_type in DETECTOR_TYPES:
            key = (profile.name, detector_type)
            if key in experts:
                f1 = experts[key]['best_trial_value']
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = experts[key]
                    best_detector_type = detector_type
        
        if best_config:
            best_experts.append(best_config)
            logger.info(f"Best expert for {profile.name}: {best_detector_type} (F1={best_f1:.4f})")
    
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
        
        phase2_csv = os.path.join(args.output_dir, "phase2_cross_dd_ensemble.csv")
        with open(phase2_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            writer.writeheader()
            writer.writerow(result)
        logger.info(f"Phase 2 results saved to {phase2_csv}")
        logger.info(f"Cross-DD best ensemble: macroF1={result['macro_f1']:.4f}")
    
    # Evaluate generalists on eval set
    logger.info("=" * 80)
    logger.info("Evaluating generalists on eval set")
    logger.info("=" * 80)
    
    generalist_eval_results = []
    for detector_type in DETECTOR_TYPES:
        if detector_type not in generalists:
            continue
        
        config = generalists[detector_type]
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
        logger.info(f"Generalist {detector_type}: macroF1={result['macro_f1']:.4f}")
    
    # Save generalist eval results
    generalist_eval_csv = os.path.join(args.output_dir, "generalists_eval.csv")
    with open(generalist_eval_csv, 'w', newline='') as f:
        if generalist_eval_results:
            writer = csv.DictWriter(f, fieldnames=generalist_eval_results[0].keys())
            writer.writeheader()
            for result in generalist_eval_results:
                writer.writerow(result)
    logger.info(f"Generalist eval results saved to {generalist_eval_csv}")
    
    logger.info("=" * 80)
    logger.info("Optimization complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
