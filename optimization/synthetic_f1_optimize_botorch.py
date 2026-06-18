"""
BoTorch-based optimization of MOPEDDS ensemble F1 on synthetic streams.

This is the BoTorch counterpart of ``synthetic_f1_optimize_optuna.py``. It
uses Ax (https://ax.dev), whose default generation strategy is BoTorch
(Sobol initialization + BoTorch GP / SAASBO models), to drive the search.

Design difference vs the Optuna script
--------------------------------------
BoTorch/Ax do not support Optuna-style runtime-conditional sampling. Per-slot
hyperparameters cannot be conditionally sampled on per-slot type without
inflating the search space by 7x. To keep BoTorch tractable up to N=128,
detector hyperparameters are **shared per detector class** (one parameter
block per class, used by every slot that selects that class). Per-slot
choices are limited to the categorical ``slot{i}_type``.

Parameters (single objective: maximize F1)
------------------------------------------
- MOPEDDS-level (5):
    detector_decision_criteria, ensemble_decision_criteria,
    decision_window, suppression_window, recent_samples_size
- Per-slot composition (N):
    slot{i}_type in {BNDM, CSDDM, D3, IBDD, OCDD, SPLL, UDetect}
- Shared per-class hyperparameters (23 total):
    bndm_*, csddm_*, d3_*, ibdd_*, ocdd_*, spll_*, udetect_*

Usage:
    python optimization/synthetic_f1_optimize_botorch.py \
        --dataset "SineClustersPre()" --size 8 --timeout 86400
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import signal
import logging
import warnings
import datetime
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

# Make repository importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the synthetic evaluation primitives.
from main_synthetic import (  # noqa: E402
    parse_expression,
    get_detector_class,
    apply_suppression,
    evaluate_detections,
    run_ensemble,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate detector pool (must match the Optuna script).
# ---------------------------------------------------------------------------

CANDIDATES = ["BNDM", "CSDDM", "D3", "IBDD", "OCDD", "SPLL", "UDetect"]

CLASS_PATH = {
    "BNDM":   "detectors.bndm.BNDM",
    "CSDDM":  "detectors.csddm.CSDDM",
    "D3":     "detectors.d3.D3",
    "IBDD":   "detectors.ibdd.IBDD",
    "OCDD":   "detectors.ocdd.OCDD",
    "SPLL":   "detectors.spll.SPLL",
    "UDetect": "detectors.udetect.UDetect",
}


# ---------------------------------------------------------------------------
# Search-space definition (Ax parameter dicts).
# ---------------------------------------------------------------------------

def _shared_detector_params():
    """Return the list of shared per-class hyperparameter specs."""
    return [
        # BNDM
        {"name": "bndm_n_samples",         "type": "range", "value_type": "int",   "bounds": [50, 500]},
        {"name": "bndm_const",             "type": "range", "value_type": "float", "bounds": [0.1, 10.0]},
        {"name": "bndm_threshold",         "type": "range", "value_type": "float", "bounds": [0.1, 0.9]},
        {"name": "bndm_max_depth",         "type": "range", "value_type": "int",   "bounds": [1, 10]},
        # CSDDM
        {"name": "csddm_n_samples",         "type": "range", "value_type": "int",   "bounds": [50, 500]},
        {"name": "csddm_feature_proportion","type": "range", "value_type": "float", "bounds": [0.1, 1.0]},
        {"name": "csddm_n_clusters",        "type": "range", "value_type": "int",   "bounds": [2, 30]},
        {"name": "csddm_confidence",        "type": "choice", "value_type": "float",
         "values": [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001], "is_ordered": True},
        # D3
        {"name": "d3_n_reference_samples",      "type": "range", "value_type": "int",   "bounds": [50, 5000]},
        {"name": "d3_recent_samples_proportion","type": "range", "value_type": "float", "bounds": [0.05, 0.5]},
        {"name": "d3_threshold",                "type": "range", "value_type": "float", "bounds": [0.1, 0.9]},
        # IBDD
        {"name": "ibdd_n_samples",                "type": "range", "value_type": "int", "bounds": [100, 2000]},
        {"name": "ibdd_n_consecutive_deviations", "type": "range", "value_type": "int", "bounds": [1, 20]},
        {"name": "ibdd_n_permutations",           "type": "range", "value_type": "int", "bounds": [100, 1000]},
        {"name": "ibdd_update_interval",          "type": "range", "value_type": "int", "bounds": [10, 100]},
        # OCDD
        {"name": "ocdd_n_samples", "type": "range", "value_type": "int",   "bounds": [50, 500]},
        {"name": "ocdd_threshold", "type": "range", "value_type": "float", "bounds": [0.1, 0.9]},
        # SPLL
        {"name": "spll_n_samples",  "type": "range", "value_type": "int",   "bounds": [100, 1000]},
        {"name": "spll_n_clusters", "type": "range", "value_type": "int",   "bounds": [2, 20]},
        {"name": "spll_threshold",  "type": "range", "value_type": "float", "bounds": [0.1, 5.0]},
        # UDetect
        {"name": "udetect_n_windows",                "type": "range",  "value_type": "int", "bounds": [5, 30]},
        {"name": "udetect_n_samples",                "type": "range",  "value_type": "int", "bounds": [20, 200]},
        {"name": "udetect_disjoint_training_windows","type": "choice", "value_type": "bool",
         "values": [True, False], "is_ordered": False},
    ]


def build_parameter_space(size: int):
    params = [
        {"name": "detector_decision_criteria", "type": "choice", "value_type": "str",
         "values": ["any", "majority", "all"], "is_ordered": False},
        {"name": "ensemble_decision_criteria", "type": "choice", "value_type": "str",
         "values": ["any", "majority", "all"], "is_ordered": False},
        {"name": "decision_window",     "type": "range", "value_type": "int", "bounds": [1, 100]},
        {"name": "suppression_window",  "type": "range", "value_type": "int", "bounds": [0, 500]},
        {"name": "recent_samples_size", "type": "range", "value_type": "int", "bounds": [50, 5000]},
    ]
    params.extend(_shared_detector_params())
    for i in range(size):
        params.append({
            "name": f"slot{i}_type",
            "type": "choice",
            "value_type": "str",
            "values": list(CANDIDATES),
            "is_ordered": False,
        })
    return params


# ---------------------------------------------------------------------------
# Detector instantiation given a flat Ax parameter dict.
# ---------------------------------------------------------------------------

def _params_for_class(p: dict, kind: str) -> dict:
    if kind == "BNDM":
        return {"n_samples": p["bndm_n_samples"], "const": p["bndm_const"],
                "threshold": p["bndm_threshold"], "max_depth": p["bndm_max_depth"]}
    if kind == "CSDDM":
        return {"n_samples": p["csddm_n_samples"],
                "feature_proportion": p["csddm_feature_proportion"],
                "n_clusters": p["csddm_n_clusters"],
                "confidence": p["csddm_confidence"]}
    if kind == "D3":
        return {"n_reference_samples": p["d3_n_reference_samples"],
                "recent_samples_proportion": p["d3_recent_samples_proportion"],
                "threshold": p["d3_threshold"]}
    if kind == "IBDD":
        return {"n_samples": p["ibdd_n_samples"],
                "n_consecutive_deviations": p["ibdd_n_consecutive_deviations"],
                "n_permutations": p["ibdd_n_permutations"],
                "update_interval": p["ibdd_update_interval"]}
    if kind == "OCDD":
        return {"n_samples": p["ocdd_n_samples"], "threshold": p["ocdd_threshold"]}
    if kind == "SPLL":
        return {"n_samples": p["spll_n_samples"],
                "n_clusters": p["spll_n_clusters"],
                "threshold": p["spll_threshold"]}
    if kind == "UDetect":
        return {"n_windows": p["udetect_n_windows"],
                "n_samples": p["udetect_n_samples"],
                "disjoint_training_windows": p["udetect_disjoint_training_windows"]}
    raise ValueError(f"Unknown detector kind: {kind}")


def _instantiate(kind: str, params: dict, seed: int, recent_samples_size: int):
    cls = get_detector_class(CLASS_PATH[kind])
    full = dict(params)
    full.setdefault("seed", seed)
    full["recent_samples_size"] = recent_samples_size
    return cls(**full)


# ---------------------------------------------------------------------------
# Trial timeout handler
# ---------------------------------------------------------------------------

class _TrialTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TrialTimeout("Trial exceeded per-trial time limit")


# ---------------------------------------------------------------------------
# Trial evaluation.
# ---------------------------------------------------------------------------

def evaluate_trial(p: dict,
                   dataset_expr: str,
                   size: int,
                   tolerance: int,
                   seed: int,
                   known_drifts,
                   per_trial_timeout: int) -> dict:
    """Run one (size, params) trial and return F1 + breakdown dict."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(per_trial_timeout)
    try:
        detectors = []
        names = []
        for i in range(size):
            kind = p[f"slot{i}_type"]
            params = _params_for_class(p, kind)
            det = _instantiate(kind, params,
                               seed=seed + i,
                               recent_samples_size=p["recent_samples_size"])
            detectors.append(det)
            names.append(f"[{i:03d}]{kind}")

        _, stream = parse_expression(dataset_expr)
        _per_member, ensemble_raw = run_ensemble(
            stream, detectors, names,
            detector_criterion=p["detector_decision_criteria"],
            ensemble_criterion=p["ensemble_decision_criteria"],
            decision_window=p["decision_window"],
        )
        ensemble_dets = apply_suppression(ensemble_raw, p["suppression_window"])
        tp, fp, fn, mean_delay = evaluate_detections(
            ensemble_dets, known_drifts, tolerance)
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return {
            "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall),
            "mean_delay": float(mean_delay) if mean_delay == mean_delay else float("nan"),
            "error": "",
        }
    except _TrialTimeout:
        logger.warning("Trial timed out")
        return {"f1": 0.0, "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "mean_delay": float("nan"),
                "error": "timeout"}
    except Exception as e:
        logger.error(f"Trial failed: {e!r}")
        return {"f1": 0.0, "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "mean_delay": float("nan"),
                "error": repr(e)}
    finally:
        signal.alarm(0)


# ---------------------------------------------------------------------------
# CSV append helper.
# ---------------------------------------------------------------------------

class _CsvAppender:
    def __init__(self, path):
        self.path = path
        self.header_written = os.path.exists(path) and os.path.getsize(path) > 0
        self.fieldnames = None
        if self.header_written:
            with open(path, "r", newline="") as f:
                self.fieldnames = csv.DictReader(f).fieldnames

    def write(self, trial_id: int, params: dict, metrics: dict):
        row = {
            "trial_id": trial_id,
            "f1": metrics["f1"],
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            "precision": metrics["precision"], "recall": metrics["recall"],
            "mean_delay": metrics["mean_delay"],
            "error": metrics["error"],
        }
        row.update(params)
        fieldnames = self.fieldnames or list(row.keys())
        for k in row.keys():
            if k not in fieldnames:
                fieldnames = list(fieldnames) + [k]
        self.fieldnames = fieldnames
        for k in fieldnames:
            row.setdefault(k, "")
        mode = "a" if self.header_written else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    ap = ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset class expression, e.g. 'SineClustersPre()'.")
    ap.add_argument("--size", type=int, required=True,
                    help="Ensemble size (number of detector slots).")
    ap.add_argument("--timeout", type=int, default=24 * 60 * 60,
                    help="Total wall-clock budget in seconds (default 86400).")
    ap.add_argument("--per-trial-timeout", type=int, default=1800,
                    help="Per-trial timeout in seconds (default 1800).")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="Optional cap on the number of trials.")
    ap.add_argument("--tolerance", type=int, default=100,
                    help="Tolerance window in samples for TP matching.")
    ap.add_argument("--seed", type=int, default=1337,
                    help="Base seed forwarded to detectors / sampler.")
    ap.add_argument("--output-dir", default="synthetic_botorch_results",
                    help="Directory for CSV / JSON snapshot files.")
    ap.add_argument("--experiment-name", default=None,
                    help="Ax experiment name (default auto: synthF1bo_<ds>_N<size>).")
    args = ap.parse_args()

    # Lazy-import Ax/BoTorch so the script's --help works without them.
    from ax.service.ax_client import AxClient, ObjectiveProperties
    from ax.storage.json_store.save import save_experiment
    from ax.storage.json_store.load import load_experiment

    dataset_label = args.dataset.split("(", 1)[0]
    os.makedirs(args.output_dir, exist_ok=True)
    results_csv = os.path.join(
        args.output_dir,
        f"synthF1bo_{dataset_label}_N{args.size}.csv")
    snapshot_path = os.path.join(
        args.output_dir,
        f"synthF1bo_{dataset_label}_N{args.size}.json")

    experiment_name = args.experiment_name or f"synthF1bo_{dataset_label}_N{args.size}"

    # Probe stream once to get known drifts.
    _, probe_stream = parse_expression(args.dataset)
    if not hasattr(probe_stream, "drifts"):
        raise ValueError(f"Dataset '{args.dataset}' has no `drifts` attribute.")
    known_drifts = list(probe_stream.drifts)

    # Set up AxClient (BoTorch is the default modeling backend).
    ax_client = AxClient(random_seed=args.seed, verbose_logging=False)

    # Load snapshot if present, otherwise create a fresh experiment.
    n_existing = 0
    if os.path.exists(snapshot_path):
        try:
            ax_client.experiment = load_experiment(snapshot_path)
            n_existing = len(ax_client.experiment.trials)
            logger.info(f"Loaded snapshot with {n_existing} existing trials")
        except Exception as e:
            logger.warning(f"Could not load snapshot ({e!r}); starting fresh")
            ax_client = AxClient(random_seed=args.seed, verbose_logging=False)

    if n_existing == 0:
        ax_client.create_experiment(
            name=experiment_name,
            parameters=build_parameter_space(args.size),
            objectives={"f1": ObjectiveProperties(minimize=False)},
            overwrite_existing_experiment=False,
        )

    appender = _CsvAppender(results_csv)

    print("=" * 80)
    print("Synthetic F1 BoTorch (Ax) Optimization")
    print("=" * 80)
    print(f"  Dataset           : {args.dataset}")
    print(f"  Ensemble size N   : {args.size}")
    print(f"  Tolerance         : {args.tolerance}")
    print(f"  Timeout (wall)    : {args.timeout}s")
    print(f"  Per-trial timeout : {args.per_trial_timeout}s")
    print(f"  Experiment        : {experiment_name}")
    print(f"  Snapshot          : {snapshot_path}")
    print(f"  Output CSV        : {results_csv}")
    print(f"  Existing trials   : {n_existing}")
    print(f"  Started           : {datetime.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80, flush=True)

    deadline = time.time() + args.timeout
    n_done = 0

    while True:
        if time.time() >= deadline:
            logger.info("Wall-clock budget exhausted")
            break
        if args.n_trials is not None and n_done >= args.n_trials:
            logger.info(f"Reached trial cap ({args.n_trials})")
            break

        try:
            params, trial_index = ax_client.get_next_trial()
        except Exception as e:
            logger.error(f"AxClient.get_next_trial failed: {e!r}; aborting")
            break

        metrics = evaluate_trial(
            params,
            dataset_expr=args.dataset,
            size=args.size,
            tolerance=args.tolerance,
            seed=args.seed,
            known_drifts=known_drifts,
            per_trial_timeout=args.per_trial_timeout,
        )

        try:
            ax_client.complete_trial(trial_index=trial_index,
                                     raw_data={"f1": (metrics["f1"], 0.0)})
        except Exception as e:
            logger.error(f"AxClient.complete_trial failed: {e!r}")

        appender.write(trial_index, params, metrics)
        n_done += 1
        logger.info(
            f"Trial {trial_index}: F1={metrics['f1']:.4f} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")

        # Persist snapshot every 10 trials so we can resume.
        if n_done % 10 == 0:
            try:
                save_experiment(ax_client.experiment, snapshot_path)
            except Exception as e:
                logger.warning(f"Snapshot save failed: {e!r}")

    # Final snapshot.
    try:
        save_experiment(ax_client.experiment, snapshot_path)
    except Exception as e:
        logger.warning(f"Final snapshot save failed: {e!r}")

    print("=" * 80)
    print(f"Optimization finished. Trials run this session: {n_done}")
    try:
        best_params, best_values = ax_client.get_best_parameters()
        if best_values is not None:
            mean = best_values[0].get("f1") if isinstance(best_values, tuple) else None
            print(f"Best F1 (model estimate): {mean}")
        print("Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"No best parameters available: {e!r}")
    print("=" * 80)


if __name__ == "__main__":
    main()
