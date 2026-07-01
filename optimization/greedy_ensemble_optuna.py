"""
Greedy Ensemble Selection using Optuna.

This script performs greedy forward-selection of MOPEDDS ensembles by running
Optuna optimization at each step to find the best detector to add to the current
ensemble.

Step 1: Load best N=1 detector from existing Optuna study
Steps 2-N: Run Optuna study (2h limit) to find best detector + MOPEDDS globals
"""

import os
import sys
import csv
import signal
import logging
import math
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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
    _run_one_stream,
    GENERATORS,
)


@dataclass
class GlobalConfig:
    detector_decision_criteria: str
    ensemble_decision_criteria: str
    decision_window: int
    suppression_window: int
    recent_samples_size: int

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default timeout for individual Optuna trials (seconds)
TRIAL_TIMEOUT = 1200  # 20 minutes


class TimeoutError(Exception):
    """Raised when a trial exceeds the timeout."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Trial exceeded timeout")


def with_timeout(func, timeout_seconds, *args, **kwargs):
    """Run a function with a timeout using signal.alarm."""
    if timeout_seconds <= 0:
        return func(*args, **kwargs)
    
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)  # Cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
    return result


@dataclass
class EnsembleMember:
    kind: str
    params: Dict[str, object]
    source: str = "optuna"


def evaluate_ensemble(*, generators: List[str],
                      drift_frequencies: List[int],
                      stream_length: int,
                      stream_seeds: List[int],
                      tolerances: List[int],
                      indices: List[int],
                      members: List[EnsembleMember],
                      global_config: GlobalConfig,
                      detector_seed: int) -> Dict[str, float]:
    """Evaluate a MOPEDDS ensemble on specified stream indices."""
    from detectors.mopedds import MOPEDDS
    from detectors.mopedds.threads_deployment import ThreadsDeployment
    
    # Build slot specs from ensemble members
    slot_specs = [(m.kind, m.params) for m in members]
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    per_stream_f1 = []
    
    for idx in indices:
        gen = generators[idx]
        drift_freq = drift_frequencies[idx]
        seed = stream_seeds[idx]
        tol = tolerances[idx]
        
        try:
            tp, fp, fn, mean_delay, f1, precision, recall, n_known = with_timeout(
                _run_one_stream,
                TRIAL_TIMEOUT,
                generator_name=gen,
                drift_frequency=drift_freq,
                stream_length=stream_length,
                stream_seed=seed,
                tolerance=tol,
                slot_specs=slot_specs,
                detector_seed_base=detector_seed,
                s_idx=idx,
                detector_criterion=global_config.detector_decision_criteria,
                ensemble_criterion=global_config.ensemble_decision_criteria,
                decision_window=global_config.decision_window,
                suppression_window=global_config.suppression_window,
                recent_samples_size=global_config.recent_samples_size,
            )
            tp_total += tp
            fp_total += fp
            fn_total += fn
            
            # Compute F1 for this stream
            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)
            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            per_stream_f1.append(f1)
        except TimeoutError:
            logger.warning(f"Evaluation timed out for stream {idx}, returning worst score")
            tp_total += 0
            fp_total += 1000  # Penalize heavily
            fn_total += 1000
            per_stream_f1.append(0.0)
    
    # Compute macro F1
    macro_f1 = sum(per_stream_f1) / len(per_stream_f1) if per_stream_f1 else 0.0
    
    return {
        "macro_f1": macro_f1,
        "per_stream_f1": per_stream_f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def suggest_detector_params(trial: optuna.Trial, detector_type: str) -> Dict[str, object]:
    """Suggest hyperparameters for a specific detector type."""
    params = {}
    
    if detector_type == "BNDM":
        params['n_samples'] = trial.suggest_int('bndm_n_samples', 50, 500)
        params['const'] = trial.suggest_float('bndm_const', 0.1, 10.0)
        params['threshold'] = trial.suggest_float('bndm_threshold', 0.1, 0.9)
        params['max_depth'] = trial.suggest_int('bndm_max_depth', 1, 10)
    elif detector_type == "CSDDM":
        params['n_samples'] = trial.suggest_int('csddm_n_samples', 50, 500)
        params['feature_proportion'] = trial.suggest_float('csddm_feature_proportion', 0.1, 1.0)
        params['n_clusters'] = trial.suggest_int('csddm_n_clusters', 2, 30)
        params['confidence'] = trial.suggest_categorical('csddm_confidence', [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    elif detector_type == "D3":
        params['n_reference_samples'] = trial.suggest_int('d3_n_reference_samples', 50, 500)
        params['recent_samples_proportion'] = trial.suggest_float('d3_recent_samples_proportion', 0.05, 0.5)
        params['threshold'] = trial.suggest_float('d3_threshold', 0.1, 0.9)
    elif detector_type == "IBDD":
        params['n_samples'] = trial.suggest_int('ibdd_n_samples', 100, 2000)
        params['n_consecutive_deviations'] = trial.suggest_int('ibdd_n_consecutive_deviations', 1, 20)
        params['n_permutations'] = trial.suggest_int('ibdd_n_permutations', 100, 1000)
        params['update_interval'] = trial.suggest_int('ibdd_update_interval', 10, 100)
    elif detector_type == "OCDD":
        params['n_samples'] = trial.suggest_int('ocdd_n_samples', 50, 500)
        params['threshold'] = trial.suggest_float('ocdd_threshold', 0.1, 0.9)
    elif detector_type == "SPLL":
        params['n_samples'] = trial.suggest_int('spll_n_samples', 100, 1000)
        params['n_clusters'] = trial.suggest_int('spll_n_clusters', 2, 20)
        params['threshold'] = trial.suggest_float('spll_threshold', 0.1, 5.0)
    elif detector_type == "UDetect":
        params['n_windows'] = trial.suggest_int('udetect_n_windows', 5, 30)
        params['n_samples'] = trial.suggest_int('udetect_n_samples', 20, 200)
        params['disjoint_training_windows'] = trial.suggest_categorical('udetect_disjoint_training_windows', [True, False])
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    return params


def create_optuna_objective(
    current_ensemble: List[EnsembleMember],
    generators: List[str],
    drift_frequencies: List[int],
    stream_length: int,
    stream_seeds: List[int],
    tolerances: List[int],
    train_indices: List[int],
    detector_seed: int,
):
    """Create Optuna objective function for greedy step."""
    
    def objective(trial: optuna.Trial) -> float:
        # Sample detector type
        detector_type = trial.suggest_categorical(
            'detector_type',
            ['IBDD', 'OCDD', 'D3', 'SPLL', 'UDetect', 'CSDDM', 'BNDM']
        )
        
        # Sample detector hyperparameters
        detector_params = suggest_detector_params(trial, detector_type)
        
        # Sample MOPEDDS global parameters
        detector_decision_criteria = trial.suggest_categorical(
            'detector_decision_criteria', ['any', 'majority', 'all'])
        ensemble_decision_criteria = trial.suggest_categorical(
            'ensemble_decision_criteria', ['any', 'majority', 'all'])
        decision_window = trial.suggest_int('decision_window', 5, 20)
        suppression_window = trial.suggest_int('suppression_window', 0, 5)
        recent_samples_size = trial.suggest_int('recent_samples_size', 100, 1000)
        
        # Create global config
        global_config = GlobalConfig(
            detector_decision_criteria=detector_decision_criteria,
            ensemble_decision_criteria=ensemble_decision_criteria,
            decision_window=decision_window,
            suppression_window=suppression_window,
            recent_samples_size=recent_samples_size,
        )
        
        # Create new ensemble with candidate
        new_ensemble = current_ensemble + [
            EnsembleMember(kind=detector_type, params=detector_params, source="optuna")
        ]
        
        # Evaluate ensemble
        try:
            metrics = evaluate_ensemble(
                generators=generators,
                drift_frequencies=drift_frequencies,
                stream_length=stream_length,
                stream_seeds=stream_seeds,
                tolerances=tolerances,
                indices=train_indices,
                members=new_ensemble,
                global_config=global_config,
                detector_seed=detector_seed,
            )
            return metrics['macro_f1']
        except TimeoutError:
            logger.warning(f"Trial timed out, returning worst score")
            return 0.0
        except Exception as e:
            logger.warning(f"Trial failed with error: {e}, returning worst score")
            return 0.0
    
    return objective


def load_best_n1_detector(pool_glob: str) -> Tuple[str, Dict[str, object], GlobalConfig]:
    """Load the best detector from CSV pool (N=1 results)."""
    import glob
    
    csv_files = glob.glob(pool_glob)
    if not csv_files:
        raise ValueError(f"No CSV files found matching glob: {pool_glob}")
    
    best_entry = None
    best_f1 = -math.inf
    
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip errored trials
                if (row.get("error") or "").strip():
                    continue
                f1 = float(row.get('macro_f1', -math.inf))
                if f1 > best_f1:
                    best_f1 = f1
                    best_entry = row
    
    if best_entry is None:
        raise ValueError(f"No valid entries found in CSV files")
    
    # Extract detector type (CSV uses slot0_type format)
    detector_type = best_entry.get('slot0_type', '')
    if not detector_type:
        raise ValueError(f"No detector type found in CSV entry")
    
    # Extract detector params (CSV uses slot0_<type>_<param> format)
    detector_params = {}
    prefix = f"slot0_{detector_type}_"
    for key, value in best_entry.items():
        if key and key.startswith(prefix):
            param_name = key[len(prefix):]
            detector_params[param_name] = value
    
    # Extract global config
    global_config = GlobalConfig(
        detector_decision_criteria=best_entry.get('detector_decision_criteria', 'majority'),
        ensemble_decision_criteria=best_entry.get('ensemble_decision_criteria', 'any'),
        decision_window=int(best_entry.get('decision_window', 10)),
        suppression_window=int(best_entry.get('suppression_window', 0)),
        recent_samples_size=int(best_entry.get('recent_samples_size', 500)),
    )
    
    logger.info(f"Loaded best N=1 detector: {detector_type} with F1={best_f1:.4f}")
    return detector_type, detector_params, global_config


def greedy_select_optuna(*,
                         pool_glob: str,
                         optuna_storage: str,
                         generators: List[str],
                         drift_frequencies: List[int],
                         stream_length: int,
                         stream_seeds: List[int],
                         tolerances: List[int],
                         train_indices: List[int],
                         eval_indices: List[int],
                         base_global: GlobalConfig,
                         detector_seed: int,
                         max_n: int,
                         n_trials: int = 50,
                         n_workers: int = 16,
                         step_timeout_hours: int = 2) -> List[Dict[str, object]]:
    """Greedy ensemble selection using Optuna at each step."""
    history: List[Dict[str, object]] = []
    ensemble: List[EnsembleMember] = []
    current_train = 0.0
    current_eval = 0.0
    base_train = 0.0
    base_eval = 0.0
    
    # Step 1: Load best N=1 detector from CSV pool
    logger.info("Step 1: Loading best N=1 detector from CSV pool")
    detector_type, detector_params, global_config = load_best_n1_detector(pool_glob)
    
    ensemble.append(EnsembleMember(kind=detector_type, params=detector_params, source="n1"))
    
    # Evaluate on train indices
    train_metrics = evaluate_ensemble(
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        indices=train_indices,
        members=ensemble,
        global_config=global_config,
        detector_seed=detector_seed,
    )
    
    # Evaluate on eval indices
    eval_metrics = evaluate_ensemble(
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        indices=eval_indices,
        members=ensemble,
        global_config=global_config,
        detector_seed=detector_seed,
    )
    
    current_train = train_metrics['macro_f1']
    current_eval = eval_metrics['macro_f1']
    base_train = current_train
    base_eval = current_eval
    
    record = {
        "step": 1,
        "n": len(ensemble),
        "added_kind": detector_type,
        "added_source": "n1",
        "added_params": detector_params,
        "ens_crit": global_config.ensemble_decision_criteria,
        "det_crit": global_config.detector_decision_criteria,
        "decision_window": global_config.decision_window,
        "suppression_window": global_config.suppression_window,
        "recent_samples_size": global_config.recent_samples_size,
        "train_macro_f1": current_train,
        "train_delta": 0.0,
        "train_improvement_vs_base": 0.0,
        "train_per_stream_f1": train_metrics['per_stream_f1'],
        "eval_macro_f1": current_eval,
        "eval_delta": 0.0,
        "eval_improvement_vs_base": 0.0,
        "eval_per_stream_f1": eval_metrics['per_stream_f1'],
        "members": [{"kind": e.kind, "source": e.source} for e in ensemble],
    }
    history.append(record)
    logger.info(
        "step=1 N=1 +%s  train_macroF1=%.4f  eval_macroF1=%.4f  det_crit=%s ens=%s dw=%d sw=%d",
        detector_type, current_train, current_eval,
        global_config.detector_decision_criteria, global_config.ensemble_decision_criteria,
        global_config.decision_window, global_config.suppression_window,
    )
    
    # Steps 2-N: Optuna search for best detector to add
    for step in range(2, max_n + 1):
        logger.info(f"Step {step}: Running Optuna study with {n_trials} trials")
        
        # Create Optuna study for this step
        study_name = f"greedy_step_{step}"
        study = optuna.create_study(
            study_name=study_name,
            storage=optuna_storage,
            sampler=TPESampler(seed=detector_seed),
            direction="maximize",
            load_if_exists=True,
        )
        
        # Create objective
        objective = create_optuna_objective(
            current_ensemble=ensemble,
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            train_indices=train_indices,
            detector_seed=detector_seed,
        )
        
        # Run optimization with timeout
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=step_timeout_hours * 3600,
                n_jobs=n_workers,
                show_progress_bar=True,
            )
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}")
            break
        
        # Get best trial
        best_trial = study.best_trial
        logger.info(f"Step {step}: Best trial F1 = {best_trial.value:.4f}")
        
        # Extract best detector and global config
        detector_type = best_trial.params['detector_type']
        detector_params = suggest_detector_params(best_trial, detector_type)
        # Re-extract params from trial
        detector_params = {k: v for k, v in best_trial.params.items() if k.startswith(detector_type.lower() + '_')}
        detector_params = {k[len(detector_type.lower()) + 1:]: v for k, v in detector_params.items()}
        
        global_config = GlobalConfig(
            detector_decision_criteria=best_trial.params['detector_decision_criteria'],
            ensemble_decision_criteria=best_trial.params['ensemble_decision_criteria'],
            decision_window=best_trial.params['decision_window'],
            suppression_window=best_trial.params['suppression_window'],
            recent_samples_size=best_trial.params['recent_samples_size'],
        )
        
        # Add to ensemble
        ensemble.append(EnsembleMember(kind=detector_type, params=detector_params, source="optuna"))
        
        # Evaluate on train indices
        train_metrics = evaluate_ensemble(
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            indices=train_indices,
            members=ensemble,
            global_config=global_config,
            detector_seed=detector_seed,
        )
        
        # Evaluate on eval indices
        eval_metrics = evaluate_ensemble(
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            indices=eval_indices,
            members=ensemble,
            global_config=global_config,
            detector_seed=detector_seed,
        )
        
        train_delta = train_metrics['macro_f1'] - current_train
        eval_delta = eval_metrics['macro_f1'] - current_eval
        train_improvement_vs_base = train_metrics['macro_f1'] - base_train
        eval_improvement_vs_base = eval_metrics['macro_f1'] - base_eval
        
        current_train = train_metrics['macro_f1']
        current_eval = eval_metrics['macro_f1']
        
        record = {
            "step": step,
            "n": len(ensemble),
            "added_kind": detector_type,
            "added_source": "optuna",
            "added_params": detector_params,
            "ens_crit": global_config.ensemble_decision_criteria,
            "det_crit": global_config.detector_decision_criteria,
            "decision_window": global_config.decision_window,
            "suppression_window": global_config.suppression_window,
            "recent_samples_size": global_config.recent_samples_size,
            "train_macro_f1": current_train,
            "train_delta": train_delta,
            "train_improvement_vs_base": train_improvement_vs_base,
            "train_per_stream_f1": train_metrics['per_stream_f1'],
            "eval_macro_f1": current_eval,
            "eval_delta": eval_delta,
            "eval_improvement_vs_base": eval_improvement_vs_base,
            "eval_per_stream_f1": eval_metrics['per_stream_f1'],
            "members": [{"kind": e.kind, "source": e.source} for e in ensemble],
        }
        history.append(record)
        logger.info(
            "step=%d N=%d +%s  train_macroF1=%.4f (%+.4f vs prev, %+.4f vs base)  eval_macroF1=%.4f (%+.4f vs prev, %+.4f vs base)  det_crit=%s ens=%s dw=%d sw=%d",
            step, len(ensemble), detector_type, current_train, train_delta, train_improvement_vs_base,
            current_eval, eval_delta, eval_improvement_vs_base,
            global_config.detector_decision_criteria, global_config.ensemble_decision_criteria,
            global_config.decision_window, global_config.suppression_window,
        )
    
    return history


def write_history_csv(path: str, history: List[Dict[str, object]]) -> None:
    """Write greedy selection history to CSV."""
    if not history:
        return
    
    fieldnames = list(history[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            # Convert complex types to strings
            row = {}
            for k, v in record.items():
                if isinstance(v, (list, dict)):
                    row[k] = str(v)
                else:
                    row[k] = v
            writer.writerow(row)


def main():
    ap = ArgumentParser(description="Greedy ensemble selection using Optuna")
    ap.add_argument("--pool-glob", required=True, help="Glob pattern for N=1 CSV pool")
    ap.add_argument("--optuna-storage", required=True, help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    ap.add_argument("--n-streams", type=int, required=True)
    ap.add_argument("--base-stream-seed", type=int, required=True)
    ap.add_argument("--drift-frequencies", type=str, required=True)
    ap.add_argument("--stream-length", type=int, required=True)
    ap.add_argument("--stream-seeds", type=str, default=None)
    ap.add_argument("--tolerances", type=str, default=None)
    ap.add_argument("--eval-stream-indices", type=str, required=True)
    ap.add_argument("--generators", type=str, default=None)
    ap.add_argument("--generator", type=str, default=None)
    ap.add_argument("--study-tag", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-n", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--step-timeout-hours", type=int, default=2)
    ap.add_argument("--globals", default="best", choices=["best", "manual"])
    ap.add_argument("--det-crit", default="majority", choices=["any", "majority", "all"])
    ap.add_argument("--ens-crit", default="any", choices=["any", "majority", "all"])
    ap.add_argument("--decision-window", type=int, default=10)
    ap.add_argument("--suppression-window", type=int, default=None)
    ap.add_argument("--recent-samples-size", type=int, default=500)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()
    
    # Resolve stream configuration
    stream_seeds = _resolve_stream_seeds(args.stream_seeds, args.n_streams, args.base_stream_seed)
    drift_frequencies = _resolve_list(args.drift_frequencies, args.n_streams, name="--drift-frequencies")
    generators_list = _resolve_generators(args.generators, args.generator, args.n_streams)
    study_tag = _resolve_study_tag(args.study_tag, generators_list)
    tolerances = (_resolve_list(args.tolerances, args.n_streams, "--tolerances")
                  if args.tolerances else _default_tolerances(drift_frequencies))
    eval_indices = sorted({int(s.strip()) for s in args.eval_stream_indices.split(',')})
    train_indices = [i for i in range(args.n_streams) if i not in eval_indices]
    
    # Resolve base global config
    if args.globals == "manual":
        suppression_window = (args.suppression_window
                              if args.suppression_window is not None
                              else max(0, min(tolerances)))
        base_global = GlobalConfig(
            detector_decision_criteria=args.det_crit,
            ensemble_decision_criteria=args.ens_crit,
            decision_window=args.decision_window,
            suppression_window=suppression_window,
            recent_samples_size=args.recent_samples_size,
        )
    else:
        # Will be loaded from N=1 study
        base_global = GlobalConfig(
            detector_decision_criteria="majority",
            ensemble_decision_criteria="any",
            decision_window=10,
            suppression_window=0,
            recent_samples_size=500,
        )
    
    print("=" * 80)
    print("Greedy ensemble selection using Optuna")
    print("=" * 80)
    print(f"  Pool glob          : {args.pool_glob}")
    print(f"  N streams          : {args.n_streams}")
    print(f"  Train indices      : {train_indices}")
    print(f"    generators       : {[generators_list[i] for i in train_indices]}")
    print(f"  Eval indices       : {eval_indices}")
    print(f"    generators       : {[generators_list[i] for i in eval_indices]}")
    print(f"  Drift frequencies  : {drift_frequencies}")
    print(f"  Max N              : {args.max_n}")
    print(f"  Trials per step    : {args.n_trials}")
    print(f"  Workers            : {args.n_workers}")
    print(f"  Step timeout       : {args.step_timeout_hours}h")
    print("=" * 80, flush=True)
    
    # Run greedy selection
    history = greedy_select_optuna(
        pool_glob=args.pool_glob,
        optuna_storage=args.optuna_storage,
        generators=generators_list,
        drift_frequencies=drift_frequencies,
        stream_length=args.stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        train_indices=train_indices,
        eval_indices=eval_indices,
        base_global=base_global,
        detector_seed=args.seed,
        max_n=args.max_n,
        n_trials=args.n_trials,
        n_workers=args.n_workers,
        step_timeout_hours=args.step_timeout_hours,
    )
    
    # Write results
    write_history_csv(args.output_csv, history)
    logger.info(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
