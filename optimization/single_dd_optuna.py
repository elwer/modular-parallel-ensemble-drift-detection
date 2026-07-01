"""
Single Drift Detector Optimization using Optuna.

This script performs optimization of a single drift detector type using Optuna,
using the same multistream evaluation setup as the joint MOPEDDS optimization.
This allows fair comparison between single DD performance and ensemble performance
with the same computational budget (500 trials).
"""

import os
import sys
import csv
import logging
from argparse import ArgumentParser
from typing import Dict, List, Tuple

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
)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from TP, FP, FN counts."""
    if tp + fp + fn == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_detector(detector_type: str, params: Dict, seed: int):
    """Get detector instance from type and parameters."""
    CLASS_PATH = {
        'IBDD': 'detectors.ibdd.IBDD',
        'OCDD': 'detectors.ocdd.OCDD',
        'D3': 'detectors.d3.D3',
        'SPLL': 'detectors.spll.SPLL',
        'UDetect': 'detectors.udetect.UDetect',
        'CSDDM': 'detectors.csddm.CSDDM',
        'BNDM': 'detectors.bndm.BNDM',
    }
    
    module_path = CLASS_PATH[detector_type]
    module_name, class_name = module_path.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    detector_class = getattr(module, class_name)
    
    return detector_class(seed=seed, **params)


def _run_one_stream(generator_name: str, drift_frequency: int, stream_length: int,
                    stream_seed: int, tolerance: int, detector_type: str,
                    detector_params: Dict, detector_seed: int, s_idx: int) -> Tuple:
    """Run single detector on a synthetic stream."""
    from datasets.sineclusters import SineClusters
    try:
        from datasets.waveform import WaveformDrift2
    except:
        WaveformDrift2 = None
    
    generator_map = {"SineClusters": SineClusters}
    if WaveformDrift2 is not None:
        generator_map["WaveformDrift2"] = WaveformDrift2
    
    generator_class = generator_map[generator_name]
    stream = generator_class(
        drift_frequency=drift_frequency,
        stream_length=stream_length,
        seed=stream_seed,
    )
    
    # Initialize detector
    det = get_detector(detector_type, detector_params, seed=detector_seed + 1000 * s_idx)
    
    # Run stream and collect detections
    detections = []
    for x, y in stream:
        if det.update(x):
            detections.append(det.drift_reported_at_sample)
    
    # Get known drift locations
    known = stream.drift_locations
    
    # Compute metrics
    tp, fp, fn, mean_delay = 0, 0, 0, 0.0
    if len(known) > 0 and len(detections) > 0:
        # Match detections to known drifts
        matched = [False] * len(known)
        for det in detections:
            matched_this = False
            for i, loc in enumerate(known):
                if not matched[i] and abs(det - loc) <= tolerance:
                    tp += 1
                    matched[i] = True
                    matched_this = True
                    mean_delay += abs(det - loc)
                    break
            if not matched_this:
                fp += 1
        fn = len([m for m in matched if not m])
        if tp > 0:
            mean_delay /= tp
    
    f1 = _f1_from_counts(tp, fp, fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return tp, fp, fn, mean_delay, f1, precision, recall, len(known)


def evaluate_detector(generators: List[str], drift_frequencies: List[int],
                     stream_length: int, stream_seeds: List[int],
                     tolerances: List[int], indices: List[int],
                     detector_type: str, detector_params: Dict,
                     detector_seed: int) -> Dict:
    """Evaluate single detector on given stream indices."""
    tp_total, fp_total, fn_total, delay_total = 0, 0, 0, 0.0
    known_total = 0
    
    for s_idx in indices:
        result = _run_one_stream(
            generator_name=generators[s_idx],
            drift_frequency=drift_frequencies[s_idx],
            stream_length=stream_length,
            stream_seed=stream_seeds[s_idx],
            tolerance=tolerances[s_idx],
            detector_type=detector_type,
            detector_params=detector_params,
            detector_seed=detector_seed,
            s_idx=s_idx,
        )
        tp, fp, fn, mean_delay, f1, precision, recall, known = result
        tp_total += tp
        fp_total += fp
        fn_total += fn
        delay_total += mean_delay * tp
        known_total += known
    
    macro_f1 = _f1_from_counts(tp_total, fp_total, fn_total)
    mean_delay = delay_total / tp_total if tp_total > 0 else 0.0
    
    return {
        'macro_f1': macro_f1,
        'mean_delay': mean_delay,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
    }


def hyperparameter_ranges(detector_type: str, trial: optuna.Trial) -> Dict:
    """Get hyperparameter ranges for a detector type."""
    if detector_type == 'IBDD':
        return {
            'n_samples': trial.suggest_int('n_samples', 500, 2000),
            'n_consecutive_deviations': trial.suggest_int('n_consecutive_deviations', 3, 20),
            'n_permutations': trial.suggest_int('n_permutations', 100, 1000),
            'update_interval': trial.suggest_int('update_interval', 5, 100),
        }
    elif detector_type == 'OCDD':
        return {
            'n_samples': trial.suggest_int('n_samples', 500, 2000),
            'threshold': trial.suggest_float('threshold', 0.1, 0.9),
        }
    elif detector_type == 'D3':
        return {
            'n_reference_samples': trial.suggest_int('n_reference_samples', 500, 2000),
            'recent_samples_proportion': trial.suggest_float('recent_samples_proportion', 0.1, 0.9),
            'threshold': trial.suggest_float('threshold', 0.1, 0.9),
        }
    elif detector_type == 'SPLL':
        return {
            'n_samples': trial.suggest_int('n_samples', 500, 2000),
            'n_clusters': trial.suggest_int('n_clusters', 2, 20),
            'threshold': trial.suggest_float('threshold', 0.1, 10.0),
        }
    elif detector_type == 'UDetect':
        return {
            'n_windows': trial.suggest_int('n_windows', 3, 10),
            'n_samples': trial.suggest_int('n_samples', 50, 500),
            'disjoint_training_windows': trial.suggest_categorical('disjoint_training_windows', [True, False]),
        }
    elif detector_type == 'CSDDM':
        return {
            'n_samples': trial.suggest_int('n_samples', 500, 2000),
            'feature_proportion': trial.suggest_float('feature_proportion', 0.5, 1.0),
            'n_clusters': trial.suggest_int('n_clusters', 2, 20),
            'confidence': trial.suggest_categorical('confidence', [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]),
        }
    elif detector_type == 'BNDM':
        return {
            'n_samples': trial.suggest_int('n_samples', 500, 2000),
            'const': trial.suggest_float('const', 0.1, 1.0),
            'threshold': trial.suggest_float('threshold', 0.1, 1.0),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
        }
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def create_optuna_objective(generators: List[str], drift_frequencies: List[int],
                           stream_length: int, stream_seeds: List[int],
                           tolerances: List[int], train_indices: List[int],
                           detector_seed: int, detector_type: str) -> callable:
    """Create Optuna objective function for single detector."""
    
    def objective(trial: optuna.Trial) -> float:
        # Sample detector hyperparameters
        params = hyperparameter_ranges(detector_type, trial)
        
        # Evaluate on training streams
        metrics = evaluate_detector(
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            indices=train_indices,
            detector_type=detector_type,
            detector_params=params,
            detector_seed=detector_seed,
        )
        
        return metrics['macro_f1']
    
    return objective


def optimize_single_dd(*,
                       optuna_storage: str,
                       generators: List[str],
                       drift_frequencies: List[int],
                       stream_length: int,
                       stream_seeds: List[int],
                       tolerances: List[int],
                       train_indices: List[int],
                       eval_indices: List[int],
                       detector_seed: int,
                       detector_type: str,
                       n_workers: int = 64,
                       n_trials: int = 500) -> Dict:
    """Single drift detector optimization using Optuna."""
    logger.info(f"Starting single DD optimization for {detector_type}")
    logger.info(f"  Storage: {optuna_storage}")
    logger.info(f"  Generators: {generators}")
    logger.info(f"  Train indices: {train_indices}")
    logger.info(f"  Eval indices: {eval_indices}")
    logger.info(f"  Workers: {n_workers}")
    logger.info(f"  Trials: {n_trials}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=f"single_{detector_type.lower()}",
        storage=optuna_storage,
        sampler=TPESampler(seed=detector_seed),
        direction="maximize",
    )
    
    objective = create_optuna_objective(
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        train_indices=train_indices,
        detector_seed=detector_seed,
        detector_type=detector_type,
    )
    
    try:
        import time
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_workers,
            show_progress_bar=True,
        )
        elapsed = time.time() - start_time
        logger.info(f"Optuna completed {n_trials} trials in {elapsed:.1f}s")
    except Exception as e:
        logger.warning(f"Optuna optimization failed: {e}")
        return {'error': str(e)}
    
    best_trial = study.best_trial
    logger.info(f"Best trial F1 = {best_trial.value:.4f}")
    
    # Reconstruct best configuration
    params = {k: v for k, v in best_trial.params.items()}
    
    # Evaluate best configuration on eval set
    eval_metrics = evaluate_detector(
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        indices=eval_indices,
        detector_type=detector_type,
        detector_params=params,
        detector_seed=detector_seed,
    )
    
    logger.info(f"Eval F1 = {eval_metrics['macro_f1']:.4f}")
    
    return {
        'detector_type': detector_type,
        'best_trial_value': best_trial.value,
        'eval_f1': eval_metrics['macro_f1'],
        'best_params': params,
    }


if __name__ == "__main__":
    ap = ArgumentParser()
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
    ap.add_argument("--detector-type", required=True, choices=['IBDD', 'OCDD', 'D3', 'SPLL', 'UDetect', 'CSDDM', 'BNDM'])
    ap.add_argument("--n-workers", type=int, default=64)
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()
    
    # Parse generators
    if args.generators:
        generators = [g.strip() for g in args.generators.split(',')]
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
    
    logger.info(f"Configuration:")
    logger.info(f"  Generators: {generators}")
    logger.info(f"  N streams: {args.n_streams}")
    logger.info(f"  Drift frequencies: {drift_frequencies}")
    logger.info(f"  Stream length: {args.stream_length}")
    logger.info(f"  Train indices: {train_indices}")
    logger.info(f"  Eval indices: {eval_indices}")
    logger.info(f"  Detector type: {args.detector_type}")
    logger.info(f"  Workers: {args.n_workers}")
    logger.info(f"  Trials: {args.n_trials}")
    
    result = optimize_single_dd(
        optuna_storage=args.optuna_storage,
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=args.stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        train_indices=train_indices,
        eval_indices=eval_indices,
        detector_seed=args.seed,
        detector_type=args.detector_type,
        n_workers=args.n_workers,
        n_trials=args.n_trials,
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['detector_type', 'best_trial_value', 'eval_f1', 'best_params'])
        writer.writeheader()
        writer.writerow(result)
    
    logger.info(f"Results saved to {args.output_csv}")
