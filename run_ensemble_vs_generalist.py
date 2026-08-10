#!/usr/bin/env python3
"""
Ensemble vs Generalist Comparison Experiment.

Budget allocation (per DD type, per fold):
  Ensemble:   m experts x n trials each  +  n deployment trials  =  m*n + n
  Generalist: m*n + n trials (same total budget)

Expected ranking:  generalist  <  per-DD ensemble  <  cross-DD ensemble

Usage:
    python run_ensemble_vs_generalist.py [--n-folds 3] [--n-budget 25] [--n-cpus 4]
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
    _suggest_detector_params,
    _instantiate,
    _f1_from_counts,
    _max_window,
    build_stream,
    MAX_WINDOW_FRACTION,
)
from main_synthetic import apply_suppression, evaluate_detections

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DETECTORS = ["SPLL", "UDetect", "D3", "OCDD", "CSDDM", "IBDD"]

DRIFT_FREQS = [200, 500, 1000]
STREAM_LENGTH = 2000
N_BUDGET = 60
N_BUDGET_MAX = 60
N_BUDGET_STEP = 15
ENSEMBLE_F1_THRESHOLD = 0.0  # disabled: fixed budget, no adaptive search
EXPERT_MIN_VAL_F1 = 0.3
N_FOLDS = 2
N_DEPLOY_TRIALS = 30
N_CPUS = 8
N_ENSEMBLE_K = 3
SEED = 1337
BASE_STREAM_SEED = 42
OUTPUT_DIR = "results_ensemble_vs_generalist"
EXPERT_TRIAL_TIMEOUT = 60
GENERALIST_TRIAL_TIMEOUT = 300

M_STREAMS = len(DRIFT_FREQS) * 2  # 6

GENERATORS_LIST: List[str] = []
DRIFT_FREQS_LIST: List[int] = []
TOLERANCES_LIST: List[int] = []
SUPPRESSIONS_LIST: List[int] = []
for freq in DRIFT_FREQS:
    GENERATORS_LIST.extend(["SineClusters", "WaveformDrift2"])
    DRIFT_FREQS_LIST.extend([freq, freq])
    TOLERANCES_LIST.extend([freq // 10, freq // 10])
    SUPPRESSIONS_LIST.extend([freq // 2, freq // 2])

N_GENERALIST_TRIALS = M_STREAMS * N_BUDGET + N_DEPLOY_TRIALS  # m*n + deploy


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

def get_stream_configs(fold: int, split: str = "train") -> List[dict]:
    seed_offset = {"train": 0, "val": 5000, "test": 10000}[split]
    return [
        {
            "generator": GENERATORS_LIST[i],
            "drift_frequency": DRIFT_FREQS_LIST[i],
            "stream_length": STREAM_LENGTH,
            "stream_seed": BASE_STREAM_SEED + seed_offset + fold * 100 + i,
            "tolerance": TOLERANCES_LIST[i],
            "suppression": SUPPRESSIONS_LIST[i],
            "stream_idx": i,
        }
        for i in range(M_STREAMS)
    ]


def _get_recent_size(params: dict) -> int:
    key = "n_samples" if "n_samples" in params else "n_reference_samples"
    return params.get(key, 200)


def _f05_from_counts(tp: int, fp: int, fn: int) -> float:
    """F_0.5 score — precision-weighted F-score.
    Favors precision 2x over recall, producing conservative detectors
    that only alert when confident (low FP)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def run_single_detector(kind, params, sc, seed, suppression=0):
    """Run a single detector on a stream.  Iterates the stream directly
    so that _TrialTimeout propagates correctly (run_ensemble in
    main_synthetic swallows all exceptions).

    suppression=0 means no suppression (raw detections).  Generalists
    are evaluated without suppression; ensembles pass their optimised
    suppression value."""
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
    dets = apply_suppression(raw, suppression) if suppression > 0 else raw
    tp, fp, fn, _ = evaluate_detections(dets, known, sc["tolerance"])
    return _f1_from_counts(tp, fp, fn), tp, fp, fn


def run_ensemble_eval(slot_specs, sc, seed,
                      det_criterion="any", ens_criterion="any",
                      decision_window=1, suppression=0):
    """Run an ensemble of detectors on a stream.  Iterates the stream
    directly so that _TrialTimeout propagates correctly.

    suppression is an optimised ensemble hyperparameter (0 = no
    suppression, otherwise the minimum gap between consecutive
    detections)."""
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
    dets = apply_suppression(raw, suppression) if suppression > 0 else raw
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


def _avail_mem_mb():
    """Return available memory in MB (Linux /proc/meminfo or psutil)."""
    try:
        import psutil
        return psutil.virtual_memory().available // (1024 * 1024)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, IndexError, ValueError):
        pass
    return None


def _clamp_workers(n_cpus, per_worker_mb=800, min_workers=2):
    """Reduce n_cpus if available memory is insufficient.
    Each worker process needs ~per_worker_mb MB (numpy, optuna, sklearn, etc.)."""
    avail_mb = _avail_mem_mb()
    if avail_mb is None:
        return n_cpus
    max_by_mem = max(min_workers, avail_mb // per_worker_mb)
    if max_by_mem < n_cpus:
        logger.info(f"  Reducing workers from {n_cpus} to {max_by_mem} "
                    f"(avail mem {avail_mb}MB, ~{per_worker_mb}MB/worker)")
        return max_by_mem
    return n_cpus


def _run_pool(worker_fn, tasks, n_cpus, max_retries=3):
    """Run tasks in a ProcessPoolExecutor with retry on BrokenProcessPool.
    Each worker handles one task then gets recycled (max_tasks_per_child=1)
    to prevent memory leaks and segfault accumulation."""
    n_cpus = _clamp_workers(n_cpus)
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

def _expert_worker(args):
    dd_type, sc, n_trials, seed, expert_idx = args
    max_window = int(sc["drift_frequency"] * MAX_WINDOW_FRACTION)
    expert_seed = seed + expert_idx * 10000

    def objective(trial):
        params = _suggest_detector_params(trial, "", dd_type, max_window=max_window)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EXPERT_TRIAL_TIMEOUT)
        try:
            f1, _, _, _ = run_single_detector(dd_type, params, sc, expert_seed)
            return f1
        except Exception:
            return 0.0
        finally:
            signal.alarm(0)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=expert_seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "dd_type": dd_type,
        "stream_idx": sc["stream_idx"],
        "expert_idx": expert_idx,
        "best_params": dict(study.best_trial.params),
        "best_train_f1": float(study.best_trial.value),
    }


def _generalist_worker(args):
    dd_type, stream_configs, n_trials, seed = args
    freqs = [c["drift_frequency"] for c in stream_configs]
    max_window = _max_window(freqs, list(range(len(freqs))))

    def objective(trial):
        params = _suggest_detector_params(trial, "", dd_type, max_window=max_window)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(GENERALIST_TRIAL_TIMEOUT)
        try:
            f1s = []
            for i, sc in enumerate(stream_configs):
                f1, _, _, _ = run_single_detector(dd_type, params, sc, seed)
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
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "dd_type": dd_type,
        "best_params": dict(study.best_trial.params),
        "best_train_f1": float(study.best_trial.value),
    }


def _deployment_worker(args):
    dd_type, per_stream_slots, stream_configs, val_configs, n_trials, seed = args

    def objective(trial):
        det_crit = trial.suggest_categorical("det_criterion", ["any", "all", "majority"])
        ens_crit = trial.suggest_categorical("ens_criterion", ["any", "all", "majority"])
        dw = trial.suggest_int("decision_window", 1, 20)
        # Suppression: ensemble-only hyperparameter, capped at freq/2 per stream
        max_supp = max(sc["suppression"] for sc in stream_configs)
        supp = trial.suggest_int("suppression", 0, max_supp)

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(GENERALIST_TRIAL_TIMEOUT)
        try:
            f1s = []
            for i, sc in enumerate(stream_configs):
                slots = per_stream_slots[sc["stream_idx"]]
                f1, _, _, _ = run_ensemble_eval(
                    slots, sc, seed, det_crit, ens_crit, dw, supp)
                f1s.append(f1)
                trial.report(sum(f1s) / len(f1s), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return sum(f1s) / len(f1s)
        except optuna.TrialPruned:
            raise
        except Exception:
            raise optuna.TrialPruned()
        finally:
            signal.alarm(0)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        completed = [study.best_trial]

    best_val_f1, best_trial = -1.0, None
    for t in completed:
        p = t.params
        val_f1s = []
        for sc in val_configs:
            slots = per_stream_slots[sc["stream_idx"]]
            f1, _, _, _ = run_ensemble_eval(
                slots, sc, seed,
                p["det_criterion"], p["ens_criterion"],
                p["decision_window"], p["suppression"])
            val_f1s.append(f1)
        val_f1 = sum(val_f1s) / len(val_f1s) if val_f1s else 0.0
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_trial = t

    return {
        "dd_type": dd_type,
        "best_params": dict(best_trial.params),
        "best_train_f1": float(best_trial.value),
        "best_val_f1": float(best_val_f1),
    }


# ============================================================
# Fold runner
# ============================================================

def _run_ensemble_pipeline(fold, n_budget, n_cpus, train_configs, val_configs, test_configs):
    """Train experts, optimize deployment, evaluate ensembles. Returns (fold_result_partial, avg_ens_f1)."""
    trials_per_expert = max(1, n_budget // N_ENSEMBLE_K)

    # ---- Phase 1: Train K experts per (DD type, stream) ----
    total_experts = M_STREAMS * len(DETECTORS) * N_ENSEMBLE_K
    logger.info(f"Phase 1: Training {total_experts} experts "
                f"({N_ENSEMBLE_K}x{trials_per_expert} trials per DD per stream)")
    t0 = time.time()
    expert_tasks = [(dd, sc, trials_per_expert, SEED, k)
                    for dd in DETECTORS
                    for sc in train_configs
                    for k in range(N_ENSEMBLE_K)]
    expert_results = _run_pool(_expert_worker, expert_tasks, n_cpus)
    experts: Dict[Tuple[str, int, int], dict] = {}
    for r in expert_results:
        experts[(r["dd_type"], r["stream_idx"], r["expert_idx"])] = r
    logger.info(f"  Experts done in {time.time()-t0:.0f}s")

    # ---- Phase 2: Per-DD ensemble deployment (K experts per stream) ----
    logger.info(f"Phase 2: Per-DD ensemble deployment optimisation "
                f"({N_DEPLOY_TRIALS} trials each, K={N_ENSEMBLE_K} experts per stream)")
    t0 = time.time()
    deploy_tasks = []
    for dd_type in DETECTORS:
        per_stream_slots = {}
        for sc in train_configs:
            s_idx = sc["stream_idx"]
            slots = [(dd_type, experts[(dd_type, s_idx, k)]["best_params"])
                     for k in range(N_ENSEMBLE_K)]
            per_stream_slots[s_idx] = slots
        deploy_tasks.append((dd_type, per_stream_slots, train_configs, val_configs,
                             N_DEPLOY_TRIALS, SEED))

    per_dd_deploy: Dict[str, dict] = {}
    deploy_results = _run_pool(_deployment_worker, deploy_tasks, n_cpus)
    for r in deploy_results:
        per_dd_deploy[r["dd_type"]] = r
        logger.info(f"  Deploy {r['dd_type']}: "
                    f"trainF1={r['best_train_f1']:.4f} valF1={r['best_val_f1']:.4f}")
    logger.info(f"  Deployment done in {time.time()-t0:.0f}s")

    # ---- Phase 3: Cross-DD ensemble (best DD type per stream on val) ----
    logger.info("Phase 3: Cross-DD ensemble selection (val F1) + deployment optimisation")
    cross_per_stream_slots = {}
    cross_dd_selection = {}
    for sc_val in val_configs:
        s_idx = sc_val["stream_idx"]
        best_f1, best_dd = -1.0, None
        for dd_type in DETECTORS:
            for k in range(N_ENSEMBLE_K):
                exp = experts.get((dd_type, s_idx, k))
                if exp:
                    f1, _, _, _ = run_single_detector(
                        dd_type, exp["best_params"], sc_val, SEED + k * 10000)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_dd = dd_type
        if best_dd:
            slots = [(best_dd, experts[(best_dd, s_idx, k)]["best_params"])
                     for k in range(N_ENSEMBLE_K)]
            cross_per_stream_slots[s_idx] = slots
            cross_dd_selection[s_idx] = {
                "dd_type": best_dd, "val_f1": best_f1,
            }
            logger.info(f"  Stream {s_idx}: best={best_dd} "
                        f"(valF1={best_f1:.4f})")

    cross_deploy = _deployment_worker(
        ("cross_dd", cross_per_stream_slots, train_configs, val_configs,
         N_DEPLOY_TRIALS, SEED))
    logger.info(f"  Cross-DD deploy: trainF1={cross_deploy['best_train_f1']:.4f} "
                f"valF1={cross_deploy['best_val_f1']:.4f}")

    # ---- Phase 4: Evaluate ensembles on test streams ----
    logger.info("Phase 4: Ensemble evaluation on test streams")

    per_dd_ensembles = {}
    good_dd_types = []
    for dd_type in DETECTORS:
        dep = per_dd_deploy[dd_type]
        if dep["best_val_f1"] < EXPERT_MIN_VAL_F1:
            logger.info(f"  Ens-{dd_type} skipped: valF1={dep['best_val_f1']:.4f} < {EXPERT_MIN_VAL_F1}")
            per_dd_ensembles[dd_type] = {
                "macro_f1": 0.0, "micro_f1": 0.0,
                "per_stream_f1": [], "tp": 0, "fp": 0, "fn": 0,
                "deployment_params": dep["best_params"],
                "filtered": True,
            }
            continue
        good_dd_types.append(dd_type)
        p = dep["best_params"]
        macro, micro, f1s, tp, fp, fn = _eval_on_streams(
            test_configs,
            lambda sc, s, _dt=dd_type, _p=p:
                run_ensemble_eval(
                    [(_dt, experts[(_dt, sc["stream_idx"], k)]["best_params"])
                     for k in range(N_ENSEMBLE_K)],
                    sc, s,
                    _p["det_criterion"], _p["ens_criterion"],
                    _p["decision_window"], _p["suppression"]),
            SEED)
        per_dd_ensembles[dd_type] = {
            "macro_f1": macro, "micro_f1": micro,
            "per_stream_f1": f1s, "tp": tp, "fp": fp, "fn": fn,
            "deployment_params": p,
        }
        logger.info(f"  Ens-{dd_type} eval: macroF1={macro:.4f}")

    # Cross-DD ensemble
    p = cross_deploy["best_params"]
    macro, micro, f1s, tp, fp, fn = _eval_on_streams(
        test_configs,
        lambda sc, s, _slots_map=cross_per_stream_slots, _p=p:
            run_ensemble_eval(
                _slots_map[sc["stream_idx"]], sc, s,
                _p["det_criterion"], _p["ens_criterion"],
                _p["decision_window"], _p["suppression"]),
        SEED)
    cross_dd_ensemble = {
        "macro_f1": macro, "micro_f1": micro,
        "per_stream_f1": f1s, "tp": tp, "fp": fp, "fn": fn,
        "deployment_params": p,
        "selection": cross_dd_selection,
    }
    logger.info(f"  Cross-DD ensemble eval: macroF1={macro:.4f}")

    avg_ens_f1 = cross_dd_ensemble["macro_f1"]
    n_good = len(good_dd_types)
    logger.info(f"  Cross-DD ensemble macroF1={avg_ens_f1:.4f} "
                f"({n_good}/{len(DETECTORS)} DD types passed val filter)")

    partial = {
        "experts": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in experts.items()},
        "per_dd_ensembles": per_dd_ensembles,
        "cross_dd_ensemble": cross_dd_ensemble,
        "good_dd_types": good_dd_types,
        "n_budget_used": n_budget,
    }
    return partial, avg_ens_f1


def run_fold(fold: int, n_budget: int, n_cpus: int) -> dict:
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold}")
    logger.info(f"{'='*70}")

    train_configs = get_stream_configs(fold, "train")
    val_configs = get_stream_configs(fold, "val")
    test_configs = get_stream_configs(fold, "test")

    # ---- Adaptive ensemble budget search ----
    current_budget = n_budget
    best_partial = None
    best_avg_f1 = -1.0

    while True:
        logger.info(f"\n--- Trying ensemble budget N_BUDGET={current_budget} "
                    f"({current_budget // N_ENSEMBLE_K} trials/expert) ---")
        partial, avg_f1 = _run_ensemble_pipeline(
            fold, current_budget, n_cpus, train_configs, val_configs, test_configs)

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_partial = partial

        if avg_f1 >= ENSEMBLE_F1_THRESHOLD:
            logger.info(f"  Cross-DD F1={avg_f1:.4f} >= {ENSEMBLE_F1_THRESHOLD} "
                        f"— threshold met, proceeding to generalist training")
            break
        elif current_budget >= N_BUDGET_MAX:
            logger.info(f"  Cross-DD F1={avg_f1:.4f} < {ENSEMBLE_F1_THRESHOLD} "
                        f"but max budget {N_BUDGET_MAX} reached — proceeding with best so far")
            break
        else:
            next_budget = min(current_budget + N_BUDGET_STEP, N_BUDGET_MAX)
            logger.info(f"  Cross-DD F1={avg_f1:.4f} < {ENSEMBLE_F1_THRESHOLD} "
                        f"— increasing budget from {current_budget} to {next_budget}")
            current_budget = next_budget

    # Use best result
    n_budget_used = best_partial["n_budget_used"]
    n_gen_trials = M_STREAMS * n_budget_used + N_DEPLOY_TRIALS

    fold_result = {
        "fold": fold,
        "n_budget_ensemble": n_budget_used,
        "ensemble_f1_threshold": ENSEMBLE_F1_THRESHOLD,
        "experts": best_partial["experts"],
        "generalists": {},
        "generalists_eval": {},
        "per_dd_ensembles": best_partial["per_dd_ensembles"],
        "cross_dd_ensemble": best_partial["cross_dd_ensemble"],
    }

    # ---- Phase 5: Train generalists and evaluate ----
    logger.info(f"\nPhase 5: Training {len(DETECTORS)} generalists "
                f"({n_gen_trials} trials each, matching ensemble budget {n_budget_used}) + evaluation")
    t0 = time.time()
    gen_tasks = [(dd, train_configs, n_gen_trials, SEED) for dd in DETECTORS]
    generalists: Dict[str, dict] = {}
    gen_results = _run_pool(_generalist_worker, gen_tasks, n_cpus)
    for r in gen_results:
        generalists[r["dd_type"]] = r
        logger.info(f"  Generalist {r['dd_type']}: "
                    f"trainF1={r['best_train_f1']:.4f}")
    logger.info(f"  Generalists done in {time.time()-t0:.0f}s")

    fold_result["generalists"] = generalists

    for dd_type in DETECTORS:
        gen = generalists[dd_type]
        macro, micro, f1s, tp, fp, fn = _eval_on_streams(
            test_configs,
            lambda sc, s, _dt=dd_type, _bp=gen["best_params"]:
                run_single_detector(_dt, _bp, sc, s),
            SEED)
        fold_result["generalists_eval"][dd_type] = {
            "macro_f1": macro, "micro_f1": micro,
            "per_stream_f1": f1s, "tp": tp, "fp": fp, "fn": fn,
        }
        logger.info(f"  Gen-{dd_type} eval: macroF1={macro:.4f}")

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

    # Per-DD ensembles
    for dd in DETECTORS:
        vals = [r["per_dd_ensembles"][dd]["macro_f1"] for r in all_results]
        mvals = [r["per_dd_ensembles"][dd]["micro_f1"] for r in all_results]
        print(f"  Ens-{dd:<29} {np.mean(vals):.4f}+/-{np.std(vals):.4f}"
              f"  {np.mean(mvals):.4f}+/-{np.std(mvals):.4f}")

    # Cross-DD
    vals = [r["cross_dd_ensemble"]["macro_f1"] for r in all_results]
    mvals = [r["cross_dd_ensemble"]["micro_f1"] for r in all_results]
    print(f"  {'Cross-DD ensemble':<33} {np.mean(vals):.4f}+/-{np.std(vals):.4f}"
          f"  {np.mean(mvals):.4f}+/-{np.std(mvals):.4f}")

    # Averages
    gen_avg = [np.mean([r["generalists_eval"][dd]["macro_f1"] for dd in DETECTORS])
               for r in all_results]
    ens_avg = [np.mean([r["per_dd_ensembles"][dd]["macro_f1"] for dd in DETECTORS])
               for r in all_results]
    cross_vals = [r["cross_dd_ensemble"]["macro_f1"] for r in all_results]

    print(f"\n  {'--- Averages ---':^65}")
    print(f"  {'Avg generalist':<33} {np.mean(gen_avg):.4f}+/-{np.std(gen_avg):.4f}")
    print(f"  {'Avg per-DD ensemble':<33} {np.mean(ens_avg):.4f}+/-{np.std(ens_avg):.4f}")
    print(f"  {'Cross-DD ensemble':<33} {np.mean(cross_vals):.4f}+/-{np.std(cross_vals):.4f}")

    best_gen = max(np.mean([r["generalists_eval"][dd]["macro_f1"] for r in all_results])
                   for dd in DETECTORS)
    best_ens = max(np.mean([r["per_dd_ensembles"][dd]["macro_f1"] for r in all_results])
                   for dd in DETECTORS)
    cross = np.mean(cross_vals)

    print(f"\n  Best generalist:       {best_gen:.4f}")
    print(f"  Best per-DD ensemble:  {best_ens:.4f}")
    print(f"  Cross-DD ensemble:     {cross:.4f}")
    if cross > best_gen:
        print(f"  *** Cross-DD ensemble wins by {cross - best_gen:.4f} ***")
    else:
        print(f"  Generalist wins by {best_gen - cross:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    global DRIFT_FREQS, STREAM_LENGTH, N_BUDGET, N_DEPLOY_TRIALS, N_FOLDS
    global EXPERT_TRIAL_TIMEOUT, GENERALIST_TRIAL_TIMEOUT, OUTPUT_DIR
    global M_STREAMS, GENERATORS_LIST, DRIFT_FREQS_LIST, TOLERANCES_LIST
    global SUPPRESSIONS_LIST, N_GENERALIST_TRIALS
    global N_BUDGET_MAX, N_BUDGET_STEP, ENSEMBLE_F1_THRESHOLD, EXPERT_MIN_VAL_F1

    ap = argparse.ArgumentParser(description="Ensemble vs Generalist Comparison")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--n-budget", type=int, default=N_BUDGET)
    ap.add_argument("--n-deploy-trials", type=int, default=N_DEPLOY_TRIALS)
    ap.add_argument("--n-cpus", type=int, default=N_CPUS)
    ap.add_argument("--drift-freqs", type=str, default=None,
                    help="Comma-separated drift frequencies (e.g. 200,500,1000)")
    ap.add_argument("--stream-length", type=int, default=None)
    ap.add_argument("--expert-timeout", type=int, default=None)
    ap.add_argument("--generalist-timeout", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    ap.add_argument("--n-budget-max", type=int, default=N_BUDGET_MAX,
                    help="Max ensemble budget for adaptive search")
    ap.add_argument("--n-budget-step", type=int, default=N_BUDGET_STEP,
                    help="Budget increment for adaptive search")
    ap.add_argument("--ensemble-f1-threshold", type=float, default=ENSEMBLE_F1_THRESHOLD,
                    help="Cross-DD ensemble F1 threshold to stop budget search")
    ap.add_argument("--expert-min-val-f1", type=float, default=EXPERT_MIN_VAL_F1,
                    help="Min val F1 for a DD type to be included in per-DD ensembles")
    args = ap.parse_args()

    if args.drift_freqs:
        DRIFT_FREQS = [int(x) for x in args.drift_freqs.split(",")]
    if args.stream_length:
        STREAM_LENGTH = args.stream_length
    N_BUDGET = args.n_budget
    N_DEPLOY_TRIALS = args.n_deploy_trials
    N_FOLDS = args.n_folds
    if args.expert_timeout:
        EXPERT_TRIAL_TIMEOUT = args.expert_timeout
    if args.generalist_timeout:
        GENERALIST_TRIAL_TIMEOUT = args.generalist_timeout
    OUTPUT_DIR = args.output_dir
    N_BUDGET_MAX = args.n_budget_max
    N_BUDGET_STEP = args.n_budget_step
    ENSEMBLE_F1_THRESHOLD = args.ensemble_f1_threshold
    EXPERT_MIN_VAL_F1 = args.expert_min_val_f1

    M_STREAMS = len(DRIFT_FREQS) * 2
    GENERATORS_LIST = []
    DRIFT_FREQS_LIST = []
    TOLERANCES_LIST = []
    SUPPRESSIONS_LIST = []
    for freq in DRIFT_FREQS:
        GENERATORS_LIST.extend(["SineClusters", "WaveformDrift2"])
        DRIFT_FREQS_LIST.extend([freq, freq])
        TOLERANCES_LIST.extend([freq // 10, freq // 10])
        SUPPRESSIONS_LIST.extend([freq // 2, freq // 2])
    N_GENERALIST_TRIALS = M_STREAMS * N_BUDGET + N_DEPLOY_TRIALS

    os.makedirs(args.output_dir, exist_ok=True)
    partial_path = os.path.join(args.output_dir, "partial.json")

    all_results = []
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        logger.info(f"Resumed from {len(all_results)} completed folds")

    for fold in range(len(all_results), args.n_folds):
        try:
            result = run_fold(fold, args.n_budget, args.n_cpus)
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
                    w.writerow([r["fold"], "per_dd_ensemble", dd,
                                r["per_dd_ensembles"][dd]["macro_f1"],
                                r["per_dd_ensembles"][dd]["micro_f1"]])
                w.writerow([r["fold"], "cross_dd_ensemble", "mixed",
                            r["cross_dd_ensemble"]["macro_f1"],
                            r["cross_dd_ensemble"]["micro_f1"]])
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
