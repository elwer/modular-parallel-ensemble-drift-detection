#!/usr/bin/env python3
"""
Greedy ensemble selection with 30-minute Optuna steps.
- Step 1: 30min Optuna to find best N=1 detector
- Step 2-N: 30min Optuna to find best additional detector
- Stop if no improvement on eval dataset
- Max N=16
"""

import argparse
import csv
import logging
import math
import multiprocessing as mp
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import optuna
from optuna.samplers import TPESampler

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.mopedds.threads_deployment import ThreadsDeployment
from datasets.sineclusters import SineClusters
try:
    from datasets.waveform import WaveformDrift2
except Exception:
    WaveformDrift2 = None
from main_synthetic import evaluate_detections, get_detector_class


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score from TP, FP, FN counts."""
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GlobalConfig:
    detector_decision_criteria: str
    ensemble_decision_criteria: str
    decision_window: int
    suppression_window: int
    recent_samples_size: int


@dataclass
class EnsembleMember:
    kind: str
    params: Dict[str, object]
    source: str


def with_timeout(func, args=(), kwargs={}, timeout=1200):
    """Run func with timeout using multiprocessing."""
    ctx = mp.get_context('spawn')
    with ctx.Pool(1) as pool:
        try:
            result = pool.apply_async(func, args, kwargs)
            return result.get(timeout=timeout)
        except mp.TimeoutError:
            logger.warning(f"Function timed out after {timeout}s")
            return None


def build_stream(generator_name: str, drift_frequency: int, stream_length: int, seed: int):
    """Build a synthetic stream."""
    generator_map = {
        'SineClusters': SineClusters,
        'WaveformDrift2': WaveformDrift2,
    }
    if generator_name not in generator_map:
        raise ValueError(f"Unknown generator: {generator_name}")
    generator_class = generator_map[generator_name]
    if generator_class is None:
        raise ValueError(f"Generator {generator_name} not available")
    return generator_class(
        n_samples=stream_length,
        drift_frequency=drift_frequency,
        seed=seed,
    )


def _run_one_stream(generator_name: str, drift_frequency: int, stream_length: int,
                    stream_seed: int, tolerance: int, slot_specs: List[Dict],
                    detector_seed_base: int, s_idx: int,
                    detector_criterion: str, ensemble_criterion: str,
                    decision_window: int, suppression_window: int,
                    recent_samples_size: int) -> Tuple:
    """Run detector on a single stream."""
    generator_map = {
        'SineClusters': SineClusters,
        'WaveformDrift2': WaveformDrift2,
    }
    if generator_name not in generator_map:
        raise ValueError(f"Unknown generator: {generator_name}")
    generator_class = generator_map[generator_name]
    if generator_class is None:
        raise ValueError(f"Generator {generator_name} not available")
    
    stream = generator_class(
        n_samples=stream_length,
        drift_frequency=drift_frequency,
        seed=stream_seed,
    )
    
    known = list(stream.drifts)
    sample_count = len(list(stream.samples))
    
    # Rebuild stream for detection
    stream = generator_class(
        n_samples=stream_length,
        drift_frequency=drift_frequency,
        seed=stream_seed,
    )
    
    detectors = []
    for spec in slot_specs:
        det = spec['detector'](
            **spec['params'],
            seed=detector_seed_base + s_idx,
        )
        detectors.append(det)
    
    mopedds = ThreadsDeployment(
        detectors=detectors,
        detector_criterion=detector_criterion,
        ensemble_criterion=ensemble_criterion,
        decision_window=decision_window,
        suppression_window=suppression_window,
        recent_samples_size=recent_samples_size,
    )
    
    detections = list(mopedds.process_stream(stream))
    
    tp, fp, fn, mean_delay = evaluate_detections(detections, known, tolerance)
    f1 = _f1_from_counts(tp, fp, fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return tp, fp, fn, mean_delay, f1, precision, recall, len(known)


def evaluate_ensemble(generators: List[str], drift_frequencies: List[int],
                     stream_length: int, stream_seeds: List[int],
                     tolerances: List[int], indices: List[int],
                     members: List[EnsembleMember], global_config: GlobalConfig,
                     detector_seed: int, per_trial_timeout: int = 900) -> Dict:
    """Evaluate ensemble on given stream indices."""
    CLASS_PATH = {
        'IBDD': 'detectors.ibdd.IBDD',
        'OCDD': 'detectors.ocdd.OCDD',
        'D3': 'detectors.d3.D3',
        'SPLL': 'detectors.spll.SPLL',
        'UDetect': 'detectors.udetect.UDetect',
        'CSDDM': 'detectors.csddm.CSDDM',
        'BNDM': 'detectors.bndm.BNDM',
    }
    
    slot_specs = []
    for member in members:
        detector_class = get_detector_class(CLASS_PATH[member.kind])
        slot_specs.append({
            'detector': detector_class,
            'params': member.params,
        })
    
    per_stream_f1 = []
    tp_total = fp_total = fn_total = 0
    
    for s_idx in indices:
        result = with_timeout(
            _run_one_stream,
            args=(
                generators[s_idx],
                drift_frequencies[s_idx],
                stream_length,
                stream_seeds[s_idx],
                tolerances[s_idx],
                slot_specs,
                detector_seed,
                s_idx,
                global_config.detector_decision_criteria,
                global_config.ensemble_decision_criteria,
                global_config.decision_window,
                global_config.suppression_window,
                global_config.recent_samples_size,
            ),
            timeout=per_trial_timeout,
        )
        if result is None:
            logger.warning(f"Stream {s_idx} evaluation timed out")
            continue
        tp, fp, fn, mean_delay, f1, prec, rec, n_known = result
        tp_total += int(tp)
        fp_total += int(fp)
        fn_total += int(fn)
        per_stream_f1.append(float(f1))
    
    macro_f1 = sum(per_stream_f1) / len(per_stream_f1) if per_stream_f1 else 0.0
    micro_f1 = tp_total / (tp_total + 0.5 * (fp_total + fn_total)) if (tp_total + fp_total + fn_total) > 0 else 0.0
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'tp_total': tp_total,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'per_stream_f1': per_stream_f1,
    }


def suggest_detector_params(trial: optuna.Trial, detector_type: str) -> Dict[str, object]:
    """Suggest detector hyperparameters based on type."""
    CLASS_PATH = {
        'IBDD': 'detectors.ibdd.IBDD',
        'OCDD': 'detectors.ocdd.OCDD',
        'D3': 'detectors.d3.D3',
        'SPLL': 'detectors.spll.SPLL',
        'UDetect': 'detectors.udetect.UDetect',
        'CSDDM': 'detectors.csddm.CSDDM',
        'BNDM': 'detectors.bndm.BNDM',
    }
    detector_class = get_detector_class(CLASS_PATH[detector_type])
    params = {}
    
    for param_name, param_range in detector_class.hyperparameter_ranges().items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            low, high = param_range
            if isinstance(low, int) and isinstance(high, int):
                params[param_name] = trial.suggest_int(f'{detector_type.lower()}_{param_name}', low, high)
            else:
                params[param_name] = trial.suggest_float(f'{detector_type.lower()}_{param_name}', low, high)
        elif isinstance(param_range, list):
            params[param_name] = trial.suggest_categorical(f'{detector_type.lower()}_{param_name}', param_range)
    
    return params


def create_optuna_objective(current_ensemble: List[EnsembleMember],
                             generators: List[str],
                             drift_frequencies: List[int],
                             stream_length: int,
                             stream_seeds: List[int],
                             tolerances: List[int],
                             train_indices: List[int],
                             detector_seed: int):
    """Create Optuna objective function."""
    
    def objective(trial: optuna.Trial) -> float:
        detector_type = trial.suggest_categorical(
            'detector_type',
            ['IBDD', 'OCDD', 'D3', 'SPLL', 'UDetect', 'CSDDM', 'BNDM']
        )
        
        detector_params = suggest_detector_params(trial, detector_type)
        
        detector_decision_criteria = trial.suggest_categorical(
            'detector_decision_criteria', ['any', 'majority', 'all'])
        ensemble_decision_criteria = trial.suggest_categorical(
            'ensemble_decision_criteria', ['any', 'majority', 'all'])
        decision_window = trial.suggest_int('decision_window', 1, 100)
        suppression_window = trial.suggest_int('suppression_window', 0, 50)
        recent_samples_size = trial.suggest_int('recent_samples_size', 100, 1000)
        
        global_config = GlobalConfig(
            detector_decision_criteria=detector_decision_criteria,
            ensemble_decision_criteria=ensemble_decision_criteria,
            decision_window=decision_window,
            suppression_window=suppression_window,
            recent_samples_size=recent_samples_size,
        )
        
        new_ensemble = current_ensemble + [EnsembleMember(
            kind=detector_type,
            params=detector_params,
            source="optuna"
        )]
        
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
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    return objective


def greedy_select_optuna(*,
                         optuna_storage: str,
                         generators: List[str],
                         drift_frequencies: List[int],
                         stream_length: int,
                         stream_seeds: List[int],
                         tolerances: List[int],
                         train_indices: List[int],
                         eval_indices: List[int],
                         detector_seed: int,
                         max_n: int = 16,
                         n_workers: int = 64,
                         step_timeout_hours: float = 0.5) -> List[Dict[str, object]]:
    """Greedy ensemble selection using Optuna at each step."""
    history: List[Dict[str, object]] = []
    ensemble: List[EnsembleMember] = []
    current_train = 0.0
    current_eval = 0.0
    base_train = 0.0
    base_eval = 0.0
    
    for step in range(1, max_n + 1):
        logger.info(f"Step {step}: Running Optuna study with {step_timeout_hours}h timeout")
        
        study_name = f"greedy_step_{step}"
        study = optuna.create_study(
            study_name=study_name,
            storage=optuna_storage,
            sampler=TPESampler(seed=detector_seed),
            direction="maximize",
            load_if_exists=True,
        )
        
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
        
        try:
            study.optimize(
                objective,
                timeout=step_timeout_hours * 3600,
                n_jobs=n_workers,
                show_progress_bar=True,
            )
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}")
            break
        
        best_trial = study.best_trial
        logger.info(f"Step {step}: Best trial F1 = {best_trial.value:.4f}")
        
        detector_type = best_trial.params['detector_type']
        detector_params = {k: v for k, v in best_trial.params.items() if k.startswith(detector_type.lower() + '_')}
        detector_params = {k[len(detector_type.lower()) + 1:]: v for k, v in detector_params.items()}
        
        global_config = GlobalConfig(
            detector_decision_criteria=best_trial.params['detector_decision_criteria'],
            ensemble_decision_criteria=best_trial.params['ensemble_decision_criteria'],
            decision_window=best_trial.params['decision_window'],
            suppression_window=best_trial.params['suppression_window'],
            recent_samples_size=best_trial.params['recent_samples_size'],
        )
        
        ensemble.append(EnsembleMember(kind=detector_type, params=detector_params, source="optuna"))
        
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
        
        if step == 1:
            base_train = train_metrics['macro_f1']
            base_eval = eval_metrics['macro_f1']
        
        train_delta = train_metrics['macro_f1'] - current_train
        eval_delta = eval_metrics['macro_f1'] - current_eval
        train_improvement_vs_base = train_metrics['macro_f1'] - base_train
        eval_improvement_vs_base = eval_metrics['macro_f1'] - base_eval
        
        logger.info(
            "step=%d N=%d +%s  train_macroF1=%.4f (%+.4f vs prev, %+.4f vs base)  eval_macroF1=%.4f (%+.4f vs prev, %+.4f vs base)  det_crit=%s ens=%s dw=%d sw=%d",
            step, len(ensemble), detector_type,
            train_metrics['macro_f1'], train_delta, train_improvement_vs_base,
            eval_metrics['macro_f1'], eval_delta, eval_improvement_vs_base,
            global_config.detector_decision_criteria, global_config.ensemble_decision_criteria,
            global_config.decision_window, global_config.suppression_window,
        )
        
        if step > 1 and eval_metrics['macro_f1'] <= current_eval + 1e-9:
            logger.warning(
                f"Step {step}: No eval improvement over current ensemble "
                f"(eval={eval_metrics['macro_f1']:.4f}, current={current_eval:.4f}); stopping."
            )
            break
        
        current_train = train_metrics['macro_f1']
        current_eval = eval_metrics['macro_f1']
        
        history.append({
            "step": step,
            "n": len(ensemble),
            "added_kind": detector_type,
            "added_params": detector_params,
            "det_crit": global_config.detector_decision_criteria,
            "ens_crit": global_config.ensemble_decision_criteria,
            "decision_window": global_config.decision_window,
            "suppression_window": global_config.suppression_window,
            "recent_samples_size": global_config.recent_samples_size,
            "train_macro_f1": train_metrics['macro_f1'],
            "train_delta": train_delta,
            "train_improvement_vs_base": train_improvement_vs_base,
            "train_per_stream_f1": train_metrics['per_stream_f1'],
            "eval_macro_f1": eval_metrics['macro_f1'],
            "eval_delta": eval_delta,
            "eval_improvement_vs_base": eval_improvement_vs_base,
            "eval_per_stream_f1": eval_metrics['per_stream_f1'],
        })
    
    return history


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--max-n", type=int, default=16)
    ap.add_argument("--n-workers", type=int, default=64)
    ap.add_argument("--step-timeout-hours", type=float, default=0.5)
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
        tolerances = [drift_frequencies[i] // 2 for i in range(args.n_streams)]
    
    # Parse eval indices
    eval_indices = sorted({int(s.strip()) for s in args.eval_stream_indices.split(',')})
    train_indices = [i for i in range(args.n_streams) if i not in eval_indices]
    
    logger.info(f"Configuration:")
    logger.info(f"  Generators: {generators}")
    logger.info(f"  Train indices: {train_indices}")
    logger.info(f"  Eval indices: {eval_indices}")
    logger.info(f"  Max N: {args.max_n}")
    logger.info(f"  Step timeout: {args.step_timeout_hours}h")
    logger.info(f"  Workers: {args.n_workers}")
    
    history = greedy_select_optuna(
        optuna_storage=args.optuna_storage,
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=args.stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        train_indices=train_indices,
        eval_indices=eval_indices,
        detector_seed=args.seed,
        max_n=args.max_n,
        n_workers=args.n_workers,
        step_timeout_hours=args.step_timeout_hours,
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        fieldnames = list(history[0].keys()) if history else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    
    logger.info(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
