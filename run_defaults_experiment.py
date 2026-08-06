#!/usr/bin/env python3
"""
Default Configs: Generalist vs Cross-DD Ensemble Experiment.

Uses the grid search ranges from Lukats et al. (2024) benchmark
(DFKI-NI config.py) to cover "default-like" configurations for each
detector. For each detector type, all grid combinations are evaluated
on training streams; the best config becomes the "generalist". The best
generalists are then combined into a cross-DD ensemble with optimised
deployment parameters.

CSDDM confidence grid includes 0.05 (the paper default) in addition to
the Lukats values [0.1, 0.01].

Usage:
    python run_defaults_experiment.py [--n-folds 3] [--n-cpus 4]
"""

import os
import sys
import json
import time
import csv
import signal
import logging
import warnings
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization.synthetic_f1_multistream_optimize_optuna import (
    _instantiate,
    _f1_from_counts,
    build_stream,
)
from main_synthetic import apply_suppression, evaluate_detections

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DETECTORS = ["SPLL", "UDetect", "D3", "OCDD", "CSDDM", "IBDD", "BNDM"]

DRIFT_FREQS = [200, 500, 1000]
STREAM_LENGTH = 2000
N_FOLDS = 3
N_DEPLOY_TRIALS = 50
N_CPUS = 4
SEED = 1337
BASE_STREAM_SEED = 42
SUPPRESSION = 50
OUTPUT_DIR = "results_defaults_experiment"
GENERALIST_TRIAL_TIMEOUT = 600

M_STREAMS = len(DRIFT_FREQS) * 2  # 2 generators per freq

GENERATORS_LIST: List[str] = []
DRIFT_FREQS_LIST: List[int] = []
TOLERANCES_LIST: List[int] = []
for freq in DRIFT_FREQS:
    GENERATORS_LIST.extend(["SineClusters", "WaveformDrift2"])
    DRIFT_FREQS_LIST.extend([freq, freq])
    TOLERANCES_LIST.extend([100, 100])


# ============================================================
# Grid definitions (from DFKI-NI config.py, Lukats et al. 2024)
# CSDDM confidence includes 0.05 (paper default) in addition to [0.1, 0.01]
# ============================================================

GRID = {
    "BNDM": {
        "n_samples": [100, 250, 500, 1000],
        "const": [0.5, 1.0, 2.0],
        "max_depth": [2, 3, 5],
        "threshold": [0.01, 0.05, 0.1],
    },
    "CSDDM": {
        "n_samples": [100, 250, 500, 1000],
        "confidence": [0.1, 0.05, 0.01],
        "feature_proportion": [0.1, 0.01],
        "n_clusters": [2, 3],
    },
    "D3": {
        "n_reference_samples": [100, 250, 500, 1000],
        "recent_samples_proportion": [0.25, 0.5, 1.0],
        "threshold": [0.01, 0.05, 0.1],
    },
    "IBDD": {
        "n_samples": [100, 250, 500, 1000],
        "n_permutations": [10, 20, 50],
        "update_interval": [50, 100, 200],
        "n_consecutive_deviations": [1, 2, 3],
    },
    "OCDD": {
        "n_samples": [100, 250, 500, 1000],
        "threshold": [0.01, 0.05, 0.1],
    },
    "SPLL": {
        "n_samples": [100, 250, 500, 1000],
        "n_clusters": [2, 3, 5],
        "threshold": [0.01, 0.05, 0.1],
    },
    "UDetect": {
        "n_windows": [5, 10, 20],
        "n_samples": [50, 100, 250],
        "disjoint_training_windows": [True, False],
    },
}


def enumerate_grid(dd_type: str) -> List[dict]:
    """Enumerate all parameter combinations for a detector type."""
    grid = GRID[dd_type]
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def grid_size(dd_type: str) -> int:
    grid = GRID[dd_type]
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


# ============================================================
# Trial timeout
# ============================================================

class _TrialTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TrialTimeout("Trial exceeded time limit")


# ============================================================
# Stream helpers
# ============================================================

def get_stream_configs(fold: int, train: bool = True) -> List[dict]:
    seed_offset = 0 if train else 10000
    return [
        {
            "generator": GENERATORS_LIST[i],
            "drift_frequency": DRIFT_FREQS_LIST[i],
            "stream_length": STREAM_LENGTH,
            "stream_seed": BASE_STREAM_SEED + seed_offset + fold * 100 + i,
            "tolerance": TOLERANCES_LIST[i],
            "stream_idx": i,
        }
        for i in range(M_STREAMS)
    ]


def _get_recent_size(params: dict) -> int:
    key = "n_samples" if "n_samples" in params else "n_reference_samples"
    return params.get(key, 200)


# ============================================================
# Evaluation primitives
# ============================================================

def run_single_detector(kind, params, sc, seed, suppression=SUPPRESSION):
    """Run a single detector on a stream."""
    stream = build_stream(sc["generator"], sc["drift_frequency"],
                          sc["stream_length"], sc["stream_seed"])
    known = list(stream.drifts)
    det = _instantiate(kind, params,
                       seed=seed + 1000 * sc["stream_idx"],
                       recent_samples_size=_get_recent_size(params))
    raw = []
    for i, (x, _y) in enumerate(stream):
        triggered = bool(det.update(x))
        if triggered:
            raw.append(i)
    dets = apply_suppression(raw, suppression)
    tp, fp, fn, _ = evaluate_detections(dets, known, sc["tolerance"])
    return _f1_from_counts(tp, fp, fn), tp, fp, fn


def run_ensemble_eval(slot_specs, sc, seed,
                      det_criterion="any", ens_criterion="any",
                      decision_window=1, suppression=SUPPRESSION):
    """Run an ensemble of detectors on a stream."""
    from collections import deque
    stream = build_stream(sc["generator"], sc["drift_frequency"],
                          sc["stream_length"], sc["stream_seed"])
    known = list(stream.drifts)
    detectors = []
    for i, (kind, params) in enumerate(slot_specs):
        det = _instantiate(kind, params,
                           seed=seed + i + 1000 * sc["stream_idx"],
                           recent_samples_size=_get_recent_size(params))
        detectors.append(det)
    histories = [deque(maxlen=max(1, decision_window)) for _ in detectors]
    raw = []
    for i, (x, _y) in enumerate(stream):
        level1 = []
        for j, det in enumerate(detectors):
            triggered = bool(det.update(x))
            histories[j].append(triggered)
            s = sum(1 for r in histories[j] if r)
            n = len(histories[j])
            if det_criterion == "any":
                level1.append(s > 0)
            elif det_criterion == "all":
                level1.append(s == n)
            else:  # majority
                level1.append(s >= (n + 1) // 2)
        ns = sum(level1)
        nd = len(level1)
        if ens_criterion == "any":
            ens = ns > 0
        elif ens_criterion == "all":
            ens = ns == nd
        else:  # majority
            ens = ns >= (nd + 1) // 2
        if ens:
            raw.append(i)
    dets = apply_suppression(raw, suppression)
    tp, fp, fn, _ = evaluate_detections(dets, known, sc["tolerance"])
    return _f1_from_counts(tp, fp, fn), tp, fp, fn


def _eval_on_streams(eval_configs, eval_fn, seed):
    f1s, tp_t, fp_t, fn_t = [], 0, 0, 0
    for sc in eval_configs:
        f1, tp, fp, fn = eval_fn(sc, seed)
        f1s.append(f1)
        tp_t += tp
        fp_t += fp
        fn_t += fn
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    micro = _f1_from_counts(tp_t, fp_t, fn_t)
    return macro, micro, f1s, tp_t, fp_t, fn_t


# ============================================================
# Robust multiprocessing
# ============================================================

def _run_pool(worker_fn, tasks, n_cpus, max_retries=3):
    """Run tasks in a ProcessPoolExecutor with retry on BrokenProcessPool.
    Each worker handles one task then gets recycled (max_tasks_per_child=1)."""
    results = []
    pending = list(tasks)
    for attempt in range(max_retries + 1):
        if not pending:
            break
        try:
            with ProcessPoolExecutor(
                max_workers=n_cpus, max_tasks_per_child=1
            ) as pool:
                futs = {pool.submit(worker_fn, t): t for t in pending}
                pending = []
                for fut in as_completed(futs):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        task = futs[fut]
                        logger.warning(f"  Task failed (attempt {attempt+1}): {e}")
                        pending.append(task)
        except BrokenProcessPool as e:
            logger.warning(f"  Pool broke (attempt {attempt+1}): {e}")
            if not pending:
                pending = [futs[f] for f in futs if not f.done()]
        if pending and attempt < max_retries:
            logger.info(f"  Retrying {len(pending)} failed tasks...")
    if pending:
        logger.error(f"  {len(pending)} tasks failed after {max_retries} retries")
    return results


# ============================================================
# Worker functions (module-level for multiprocessing)
# ============================================================

def _grid_worker(args):
    """Run a single grid config on all training streams. Returns mean F1."""
    dd_type, params, stream_configs, seed = args
    f1s = []
    for sc in stream_configs:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(GENERALIST_TRIAL_TIMEOUT)
        try:
            f1, _, _, _ = run_single_detector(dd_type, params, sc, seed)
            f1s.append(f1)
        except Exception:
            f1s.append(0.0)
        finally:
            signal.alarm(0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "dd_type": dd_type,
        "params": params,
        "train_macro_f1": macro_f1,
        "per_stream_f1": f1s,
    }


def _deployment_worker(args):
    """Optimise ensemble deployment params via Optuna."""
    slot_specs, stream_configs, n_trials, seed = args

    def objective(trial):
        det_crit = trial.suggest_categorical("det_criterion", ["any", "all", "majority"])
        ens_crit = trial.suggest_categorical("ens_criterion", ["any", "all", "majority"])
        dw = trial.suggest_int("decision_window", 1, 20)
        supp = trial.suggest_int("suppression", 0, 200)

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(GENERALIST_TRIAL_TIMEOUT)
        try:
            f1s = []
            for i, sc in enumerate(stream_configs):
                f1, _, _, _ = run_ensemble_eval(
                    slot_specs, sc, seed, det_crit, ens_crit, dw, supp)
                f1s.append(f1)
                trial.report(sum(f1s) / len(f1s), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return sum(f1s) / len(f1s)
        except Exception:
            return 0.0
        finally:
            signal.alarm(0)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": dict(study.best_trial.params),
        "best_train_f1": float(study.best_trial.value),
    }


# ============================================================
# Fold runner
# ============================================================

def run_fold(fold: int, n_cpus: int) -> dict:
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold}")
    logger.info(f"{'='*70}")

    train_configs = get_stream_configs(fold, train=True)
    eval_configs = get_stream_configs(fold, train=False)

    # ---- Phase 1: Grid search for best generalist per DD type ----
    logger.info("Phase 1: Grid search for best default-like generalist configs")
    generalists: Dict[str, dict] = {}

    for dd_type in DETECTORS:
        combos = enumerate_grid(dd_type)
        logger.info(f"  {dd_type}: {len(combos)} grid combinations")

        tasks = [(dd_type, params, train_configs, SEED) for params in combos]
        t0 = time.time()
        results = _run_pool(_grid_worker, tasks, n_cpus)

        best = max(results, key=lambda r: r["train_macro_f1"])
        generalists[dd_type] = best
        logger.info(f"  {dd_type} best: trainF1={best['train_macro_f1']:.4f} "
                    f"params={best['params']} ({time.time()-t0:.0f}s)")

    # ---- Phase 2: Cross-DD ensemble deployment optimisation ----
    logger.info(f"Phase 2: Cross-DD ensemble deployment optimisation "
                f"({N_DEPLOY_TRIALS} trials)")
    cross_dd_slots = [(dd_type, generalists[dd_type]["params"])
                      for dd_type in DETECTORS]
    t0 = time.time()
    cross_deploy = _deployment_worker(
        (cross_dd_slots, train_configs, N_DEPLOY_TRIALS, SEED))
    logger.info(f"  Cross-DD deploy: trainF1={cross_deploy['best_train_f1']:.4f} "
                f"({time.time()-t0:.0f}s)")

    # ---- Phase 3: Evaluate on held-out streams ----
    logger.info("Phase 3: Evaluation on held-out streams")

    fold_result = {
        "fold": fold,
        "generalists": {},
        "generalists_eval": {},
        "cross_dd_ensemble": {},
    }

    # Store generalist details
    for dd_type in DETECTORS:
        g = generalists[dd_type]
        fold_result["generalists"][dd_type] = {
            "params": g["params"],
            "train_macro_f1": g["train_macro_f1"],
            "train_per_stream_f1": g["per_stream_f1"],
        }

    # Evaluate generalists
    for dd_type in DETECTORS:
        params = generalists[dd_type]["params"]
        macro, micro, f1s, tp, fp, fn = _eval_on_streams(
            eval_configs,
            lambda sc, s, _dt=dd_type, _bp=params:
                run_single_detector(_dt, _bp, sc, s),
            SEED)
        fold_result["generalists_eval"][dd_type] = {
            "macro_f1": macro, "micro_f1": micro,
            "per_stream_f1": f1s, "tp": tp, "fp": fp, "fn": fn,
        }
        logger.info(f"  Gen-{dd_type} eval: macroF1={macro:.4f}")

    # Evaluate cross-DD ensemble
    p = cross_deploy["best_params"]
    macro, micro, f1s, tp, fp, fn = _eval_on_streams(
        eval_configs,
        lambda sc, s, _slots=cross_dd_slots, _p=p:
            run_ensemble_eval(_slots, sc, s,
                              _p["det_criterion"], _p["ens_criterion"],
                              _p["decision_window"], _p["suppression"]),
        SEED)
    fold_result["cross_dd_ensemble"] = {
        "macro_f1": macro, "micro_f1": micro,
        "per_stream_f1": f1s, "tp": tp, "fp": fp, "fn": fn,
        "deployment_params": p,
    }
    logger.info(f"  Cross-DD ensemble eval: macroF1={macro:.4f}")

    return fold_result


# ============================================================
# Comparison table
# ============================================================

def print_comparison_table(all_results: list):
    n = len(all_results)
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON  (mean +/- std across {n} folds)")
    print(f"{'='*80}")
    print(f"{'Approach':<35} {'Macro F1':>15} {'Micro F1':>15}")
    print(f"{'-'*35} {'-'*15} {'-'*15}")

    # Generalists
    for dd in DETECTORS:
        vals = [r["generalists_eval"][dd]["macro_f1"] for r in all_results]
        mvals = [r["generalists_eval"][dd]["micro_f1"] for r in all_results]
        print(f"  Gen-{dd:<29} {np.mean(vals):.4f}+/-{np.std(vals):.4f}"
              f"  {np.mean(mvals):.4f}+/-{np.std(mvals):.4f}")

    # Cross-DD ensemble
    vals = [r["cross_dd_ensemble"]["macro_f1"] for r in all_results]
    mvals = [r["cross_dd_ensemble"]["micro_f1"] for r in all_results]
    print(f"  {'Cross-DD ensemble':<33} {np.mean(vals):.4f}+/-{np.std(vals):.4f}"
          f"  {np.mean(mvals):.4f}+/-{np.std(mvals):.4f}")

    # Averages
    gen_avg = [np.mean([r["generalists_eval"][dd]["macro_f1"] for dd in DETECTORS])
               for r in all_results]
    cross_vals = [r["cross_dd_ensemble"]["macro_f1"] for r in all_results]

    print(f"\n  {'--- Averages ---':^65}")
    print(f"  {'Avg generalist':<33} {np.mean(gen_avg):.4f}+/-{np.std(gen_avg):.4f}")
    print(f"  {'Cross-DD ensemble':<33} {np.mean(cross_vals):.4f}+/-{np.std(cross_vals):.4f}")

    best_gen = max(np.mean([r["generalists_eval"][dd]["macro_f1"] for r in all_results])
                   for dd in DETECTORS)
    cross = np.mean(cross_vals)

    print(f"\n  Best generalist:       {best_gen:.4f}")
    print(f"  Cross-DD ensemble:     {cross:.4f}")
    if cross > best_gen:
        print(f"  *** Cross-DD ensemble wins by {cross - best_gen:.4f} ***")
    else:
        print(f"  Generalist wins by {best_gen - cross:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    global DRIFT_FREQS, STREAM_LENGTH, N_FOLDS, N_DEPLOY_TRIALS
    global GENERALIST_TRIAL_TIMEOUT, OUTPUT_DIR
    global M_STREAMS, GENERATORS_LIST, DRIFT_FREQS_LIST, TOLERANCES_LIST

    ap = argparse.ArgumentParser(
        description="Default Configs: Generalist vs Cross-DD Ensemble")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--n-deploy-trials", type=int, default=N_DEPLOY_TRIALS)
    ap.add_argument("--n-cpus", type=int, default=N_CPUS)
    ap.add_argument("--drift-freqs", type=str, default=None,
                    help="Comma-separated drift frequencies (e.g. 200,500,1000)")
    ap.add_argument("--stream-length", type=int, default=None)
    ap.add_argument("--generalist-timeout", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = ap.parse_args()

    if args.drift_freqs:
        DRIFT_FREQS = [int(x) for x in args.drift_freqs.split(",")]
    if args.stream_length:
        STREAM_LENGTH = args.stream_length
    N_FOLDS = args.n_folds
    N_DEPLOY_TRIALS = args.n_deploy_trials
    if args.generalist_timeout:
        GENERALIST_TRIAL_TIMEOUT = args.generalist_timeout
    OUTPUT_DIR = args.output_dir

    M_STREAMS = len(DRIFT_FREQS) * 2
    GENERATORS_LIST = []
    DRIFT_FREQS_LIST = []
    TOLERANCES_LIST = []
    for freq in DRIFT_FREQS:
        GENERATORS_LIST.extend(["SineClusters", "WaveformDrift2"])
        DRIFT_FREQS_LIST.extend([freq, freq])
        TOLERANCES_LIST.extend([100, 100])

    # Log grid sizes
    total_configs = 0
    for dd in DETECTORS:
        sz = grid_size(dd)
        total_configs += sz
        logger.info(f"  {dd}: {sz} grid configs")
    logger.info(f"Total grid configs across all detectors: {total_configs}")

    os.makedirs(args.output_dir, exist_ok=True)
    partial_path = os.path.join(args.output_dir, "partial.json")

    all_results = []
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        logger.info(f"Resumed from {len(all_results)} completed folds")

    for fold in range(len(all_results), args.n_folds):
        try:
            result = run_fold(fold, args.n_cpus)
            all_results.append(result)
            with open(partial_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Fold {fold} complete and saved")
        except Exception as e:
            logger.error(f"Fold {fold} failed: {e}")
            import traceback
            traceback.print_exc()
            break

    if all_results:
        print_comparison_table(all_results)

        summary_path = os.path.join(args.output_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "approach", "dd_type", "macro_f1", "micro_f1"])
            for r in all_results:
                for dd in DETECTORS:
                    w.writerow([r["fold"], "generalist", dd,
                                r["generalists_eval"][dd]["macro_f1"],
                                r["generalists_eval"][dd]["micro_f1"]])
                w.writerow([r["fold"], "cross_dd_ensemble", "mixed",
                            r["cross_dd_ensemble"]["macro_f1"],
                            r["cross_dd_ensemble"]["micro_f1"]])
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
