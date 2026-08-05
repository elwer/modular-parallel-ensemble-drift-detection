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

DETECTORS = ["SPLL", "UDetect", "D3", "OCDD", "CSDDM", "IBDD", "BNDM"]

DRIFT_FREQS = [200, 500, 1000]
STREAM_LENGTH = 2000
N_BUDGET = 15
N_FOLDS = 2
N_DEPLOY_TRIALS = 25
N_CPUS = 4
SEED = 1337
BASE_STREAM_SEED = 42
SUPPRESSION = 50
OUTPUT_DIR = "results_ensemble_vs_generalist"
EXPERT_TRIAL_TIMEOUT = 60
GENERALIST_TRIAL_TIMEOUT = 300

M_STREAMS = len(DRIFT_FREQS) * 2  # 10

GENERATORS_LIST: List[str] = []
DRIFT_FREQS_LIST: List[int] = []
TOLERANCES_LIST: List[int] = []
for freq in DRIFT_FREQS:
    GENERATORS_LIST.extend(["SineClusters", "WaveformDrift2"])
    DRIFT_FREQS_LIST.extend([freq, freq])
    TOLERANCES_LIST.extend([100, 100])

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


def _f05_from_counts(tp: int, fp: int, fn: int) -> float:
    """F_0.5 score — precision-weighted F-score.
    Favors precision 2x over recall, producing conservative detectors
    that only alert when confident (low FP)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def run_single_detector(kind, params, sc, seed, suppression=SUPPRESSION):
    """Run a single detector on a stream.  Iterates the stream directly
    so that _TrialTimeout propagates correctly (run_ensemble in
    main_synthetic swallows all exceptions)."""
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
    """Run an ensemble of detectors on a stream.  Iterates the stream
    directly so that _TrialTimeout propagates correctly."""
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
# Worker functions (module-level for multiprocessing)
# ============================================================

def _expert_worker(args):
    dd_type, sc, n_trials, seed = args
    max_window = int(sc["drift_frequency"] * MAX_WINDOW_FRACTION)

    def objective(trial):
        params = _suggest_detector_params(trial, "", dd_type, max_window=max_window)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EXPERT_TRIAL_TIMEOUT)
        try:
            f1, _, _, _ = run_single_detector(dd_type, params, sc, seed)
            return f1
        except Exception:
            return 0.0
        finally:
            signal.alarm(0)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "dd_type": dd_type,
        "stream_idx": sc["stream_idx"],
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
    dd_type, slot_specs, stream_configs, n_trials, seed = args

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
        "dd_type": dd_type,
        "best_params": dict(study.best_trial.params),
        "best_train_f1": float(study.best_trial.value),
    }


# ============================================================
# Fold runner
# ============================================================

def run_fold(fold: int, n_budget: int, n_cpus: int) -> dict:
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold}")
    logger.info(f"{'='*70}")

    train_configs = get_stream_configs(fold, train=True)
    eval_configs = get_stream_configs(fold, train=False)
    n_gen_trials = M_STREAMS * n_budget + N_DEPLOY_TRIALS

    # ---- Phase 1: Train experts (m streams x 7 DD types) ----
    logger.info(f"Phase 1: Training {M_STREAMS * len(DETECTORS)} experts "
                f"({n_budget} trials each)")
    t0 = time.time()
    expert_tasks = [(dd, sc, n_budget, SEED)
                    for dd in DETECTORS for sc in train_configs]
    experts: Dict[Tuple[str, int], dict] = {}
    with ProcessPoolExecutor(max_workers=n_cpus) as pool:
        futs = {pool.submit(_expert_worker, t): t for t in expert_tasks}
        for fut in as_completed(futs):
            r = fut.result()
            experts[(r["dd_type"], r["stream_idx"])] = r
    logger.info(f"  Experts done in {time.time()-t0:.0f}s")

    # ---- Phase 2: Train generalists (7 DD types, m*n+n trials each) ----
    logger.info(f"Phase 2: Training {len(DETECTORS)} generalists "
                f"({n_gen_trials} trials each)")
    t0 = time.time()
    gen_tasks = [(dd, train_configs, n_gen_trials, SEED) for dd in DETECTORS]
    generalists: Dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=n_cpus) as pool:
        futs = {pool.submit(_generalist_worker, t): t for t in gen_tasks}
        for fut in as_completed(futs):
            r = fut.result()
            generalists[r["dd_type"]] = r
            logger.info(f"  Generalist {r['dd_type']}: "
                        f"trainF1={r['best_train_f1']:.4f}")
    logger.info(f"  Generalists done in {time.time()-t0:.0f}s")

    # ---- Phase 3: Optimise deployment params for per-DD ensembles ----
    logger.info(f"Phase 3: Per-DD ensemble deployment optimisation "
                f"({N_DEPLOY_TRIALS} trials each)")
    t0 = time.time()
    deploy_tasks = []
    for dd_type in DETECTORS:
        slots = [(dd_type, experts[(dd_type, sc["stream_idx"])]["best_params"])
                 for sc in train_configs]
        deploy_tasks.append((dd_type, slots, train_configs, N_DEPLOY_TRIALS, SEED))

    per_dd_deploy: Dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=n_cpus) as pool:
        futs = {pool.submit(_deployment_worker, t): t for t in deploy_tasks}
        for fut in as_completed(futs):
            r = fut.result()
            per_dd_deploy[r["dd_type"]] = r
            logger.info(f"  Deploy {r['dd_type']}: "
                        f"trainF1={r['best_train_f1']:.4f}")
    logger.info(f"  Deployment done in {time.time()-t0:.0f}s")

    # ---- Phase 4: Cross-DD ensemble selection + deployment ----
    logger.info("Phase 4: Cross-DD ensemble selection + deployment optimisation")
    cross_dd_slots: List[Tuple[str, dict]] = []
    cross_dd_selection = {}
    for sc in train_configs:
        best_f1, best_dd, best_p = -1.0, None, None
        for dd_type in DETECTORS:
            exp = experts.get((dd_type, sc["stream_idx"]))
            if exp and exp["best_train_f1"] > best_f1:
                best_f1 = exp["best_train_f1"]
                best_dd = dd_type
                best_p = exp["best_params"]
        if best_dd:
            cross_dd_slots.append((best_dd, best_p))
            cross_dd_selection[sc["stream_idx"]] = {
                "dd_type": best_dd, "train_f1": best_f1,
            }
            logger.info(f"  Stream {sc['stream_idx']}: best={best_dd} "
                        f"(trainF1={best_f1:.4f})")

    cross_deploy = _deployment_worker(
        ("cross_dd", cross_dd_slots, train_configs, N_DEPLOY_TRIALS, SEED))
    logger.info(f"  Cross-DD deploy: trainF1={cross_deploy['best_train_f1']:.4f}")

    # ---- Phase 5: Evaluate on held-out streams ----
    logger.info("Phase 5: Evaluation on held-out streams")

    fold_result = {
        "fold": fold,
        "experts": {f"{k[0]}_{k[1]}": v for k, v in experts.items()},
        "generalists": generalists,
        "generalists_eval": {},
        "per_dd_ensembles": {},
        "cross_dd_ensemble": {},
    }

    # Generalists
    for dd_type in DETECTORS:
        gen = generalists[dd_type]
        macro, micro, f1s, tp, fp, fn = _eval_on_streams(
            eval_configs,
            lambda sc, s, _dt=dd_type, _bp=gen["best_params"]:
                run_single_detector(_dt, _bp, sc, s),
            SEED)
        fold_result["generalists_eval"][dd_type] = {
            "macro_f1": macro, "micro_f1": micro,
            "per_stream_f1": f1s, "tp": tp, "fp": fp, "fn": fn,
        }
        logger.info(f"  Gen-{dd_type} eval: macroF1={macro:.4f}")

    # Per-DD ensembles
    for dd_type in DETECTORS:
        dep = per_dd_deploy[dd_type]
        slots = [(dd_type, experts[(dd_type, sc["stream_idx"])]["best_params"])
                 for sc in train_configs]
        p = dep["best_params"]
        macro, micro, f1s, tp, fp, fn = _eval_on_streams(
            eval_configs,
            lambda sc, s, _slots=slots, _p=p:
                run_ensemble_eval(_slots, sc, s,
                                  _p["det_criterion"], _p["ens_criterion"],
                                  _p["decision_window"], _p["suppression"]),
            SEED)
        fold_result["per_dd_ensembles"][dd_type] = {
            "macro_f1": macro, "micro_f1": micro,
            "per_stream_f1": f1s, "tp": tp, "fp": fp, "fn": fn,
            "deployment_params": p,
        }
        logger.info(f"  Ens-{dd_type} eval: macroF1={macro:.4f}")

    # Cross-DD ensemble
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
        "selection": cross_dd_selection,
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
    global N_GENERALIST_TRIALS

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

    M_STREAMS = len(DRIFT_FREQS) * 2
    GENERATORS_LIST = []
    DRIFT_FREQS_LIST = []
    TOLERANCES_LIST = []
    for freq in DRIFT_FREQS:
        GENERATORS_LIST.extend(["SineClusters", "WaveformDrift2"])
        DRIFT_FREQS_LIST.extend([freq, freq])
        TOLERANCES_LIST.extend([100, 100])
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
