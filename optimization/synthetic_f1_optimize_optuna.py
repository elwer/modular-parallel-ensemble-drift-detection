"""
Optuna optimization of MOPEDDS ensemble F1 on synthetic streams.

For a fixed (dataset, ensemble_size), Optuna jointly optimizes:
  - MOPEDDS-level parameters
        detector_decision_criteria, ensemble_decision_criteria,
        decision_window, suppression_window, recent_samples_size
  - Per-slot ensemble composition (which detector class is in each slot)
        slot{i}_type in {BNDM, CSDDM, D3, IBDD, OCDD, SPLL, UDetect}
  - Per-slot detector hyperparameters (sampled conditionally on the slot type)

The objective is the F1 score of the ENSEMBLE detections against the known
drift points of the synthetic stream:
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        F1        = 2 * precision * recall / (precision + recall)

Single-objective: maximize F1.

Usage:
    python optimization/synthetic_f1_optimize_optuna.py \
        --dataset "SineClustersPre()" --size 8 --timeout 86400
"""

from __future__ import annotations

import os
import sys
import csv
import signal
import logging
import warnings
import datetime
from argparse import ArgumentParser

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

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
# Candidate detector pool
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


def _suggest_detector_params(trial: optuna.Trial, prefix: str, kind: str) -> dict:
    """Suggest hyperparameters for a single detector slot.

    Param ranges mirror ``optimization/single_dd_optimize_optuna.py`` /
    ``generate_scalability_configs.py``.
    """
    p = prefix
    if kind == "BNDM":
        return {
            "n_samples": trial.suggest_int(f"{p}n_samples", 50, 500),
            "const": trial.suggest_float(f"{p}const", 0.1, 10.0),
            "threshold": trial.suggest_float(f"{p}threshold", 0.1, 0.9),
            "max_depth": trial.suggest_int(f"{p}max_depth", 1, 10),
        }
    if kind == "CSDDM":
        return {
            "n_samples": trial.suggest_int(f"{p}n_samples", 50, 500),
            "feature_proportion": trial.suggest_float(f"{p}feature_proportion", 0.1, 1.0),
            "n_clusters": trial.suggest_int(f"{p}n_clusters", 2, 30),
            "confidence": trial.suggest_categorical(
                f"{p}confidence", [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]),
        }
    if kind == "D3":
        return {
            "n_reference_samples": trial.suggest_int(f"{p}n_reference_samples", 50, 5000),
            "recent_samples_proportion": trial.suggest_float(f"{p}recent_samples_proportion", 0.05, 0.5),
            "threshold": trial.suggest_float(f"{p}threshold", 0.1, 0.9),
        }
    if kind == "IBDD":
        return {
            "n_samples": trial.suggest_int(f"{p}n_samples", 100, 2000),
            "n_consecutive_deviations": trial.suggest_int(f"{p}n_consecutive_deviations", 1, 20),
            "n_permutations": trial.suggest_int(f"{p}n_permutations", 100, 1000),
            "update_interval": trial.suggest_int(f"{p}update_interval", 10, 100),
        }
    if kind == "OCDD":
        return {
            "n_samples": trial.suggest_int(f"{p}n_samples", 50, 500),
            "threshold": trial.suggest_float(f"{p}threshold", 0.1, 0.9),
        }
    if kind == "SPLL":
        return {
            "n_samples": trial.suggest_int(f"{p}n_samples", 100, 1000),
            "n_clusters": trial.suggest_int(f"{p}n_clusters", 2, 20),
            "threshold": trial.suggest_float(f"{p}threshold", 0.1, 5.0),
        }
    if kind == "UDetect":
        return {
            "n_windows": trial.suggest_int(f"{p}n_windows", 5, 30),
            "n_samples": trial.suggest_int(f"{p}n_samples", 20, 200),
            "disjoint_training_windows": trial.suggest_categorical(
                f"{p}disjoint_training_windows", [True, False]),
        }
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
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(dataset_expr: str,
                   size: int,
                   tolerance: int,
                   seed: int,
                   per_trial_timeout: int):
    # Probe stream once to get known drifts and stream length.
    _, probe_stream = parse_expression(dataset_expr)
    if not hasattr(probe_stream, "drifts"):
        raise ValueError(f"Dataset '{dataset_expr}' has no `drifts` attribute.")
    known_drifts = list(probe_stream.drifts)
    n_known = len(known_drifts)
    logger.info(f"Dataset={dataset_expr} known_drifts={n_known} size={size}")

    def objective(trial: optuna.Trial) -> float:
        # MOPEDDS-level params
        det_crit = trial.suggest_categorical(
            "detector_decision_criteria", ["any", "majority", "all"])
        ens_crit = trial.suggest_categorical(
            "ensemble_decision_criteria", ["any", "majority", "all"])
        decision_window = trial.suggest_int("decision_window", 1, 100)
        suppression_window = trial.suggest_int("suppression_window", 0, 500)
        recent_samples_size = trial.suggest_int("recent_samples_size", 50, 5000)

        # Slot composition + per-slot params
        slot_specs = []
        for i in range(size):
            kind = trial.suggest_categorical(f"slot{i}_type", CANDIDATES)
            params = _suggest_detector_params(trial, f"slot{i}_{kind}_", kind)
            slot_specs.append((kind, params))

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(per_trial_timeout)

        try:
            detectors = []
            names = []
            for i, (kind, params) in enumerate(slot_specs):
                det = _instantiate(kind, params,
                                   seed=seed + i,
                                   recent_samples_size=recent_samples_size)
                detectors.append(det)
                names.append(f"[{i:03d}]{kind}")

            # Fresh stream iterator per trial.
            _, stream = parse_expression(dataset_expr)
            _per_member, ensemble_raw = run_ensemble(
                stream, detectors, names,
                detector_criterion=det_crit,
                ensemble_criterion=ens_crit,
                decision_window=decision_window,
            )

            ensemble_dets = apply_suppression(ensemble_raw, suppression_window)
            tp, fp, fn, mean_delay = evaluate_detections(
                ensemble_dets, known_drifts, tolerance)

            denom = (2 * tp + fp + fn)
            f1 = (2 * tp) / denom if denom > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            trial.set_user_attr("tp", int(tp))
            trial.set_user_attr("fp", int(fp))
            trial.set_user_attr("fn", int(fn))
            trial.set_user_attr("precision", float(precision))
            trial.set_user_attr("recall", float(recall))
            trial.set_user_attr("mean_delay",
                                float(mean_delay) if mean_delay == mean_delay else float("nan"))

            logger.info(
                f"Trial {trial.number}: F1={f1:.4f} P={precision:.3f} R={recall:.3f} "
                f"TP={tp} FP={fp} FN={fn}")
            return float(f1)

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

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _csv_writer(path):
    """Return a callback that appends each completed trial to `path`."""
    header_written = os.path.exists(path) and os.path.getsize(path) > 0
    existing_fieldnames = None
    if header_written:
        with open(path, "r", newline="") as f:
            existing_fieldnames = csv.DictReader(f).fieldnames

    def callback(study, trial):
        nonlocal header_written, existing_fieldnames
        if trial.state != TrialState.COMPLETE:
            return
        row = {
            "trial_id": trial.number,
            "f1": trial.value,
            "tp": trial.user_attrs.get("tp", ""),
            "fp": trial.user_attrs.get("fp", ""),
            "fn": trial.user_attrs.get("fn", ""),
            "precision": trial.user_attrs.get("precision", ""),
            "recall": trial.user_attrs.get("recall", ""),
            "mean_delay": trial.user_attrs.get("mean_delay", ""),
            "error": trial.user_attrs.get("error", ""),
        }
        row.update(trial.params)
        fieldnames = existing_fieldnames or list(row.keys())
        # Ensure any new keys are appended at the end (Optuna conditional
        # params introduce different columns across trials).
        for k in row.keys():
            if k not in fieldnames:
                fieldnames = list(fieldnames) + [k]
        existing_fieldnames = fieldnames
        # Pad missing keys to "" so DictWriter does not raise.
        for k in fieldnames:
            row.setdefault(k, "")
        mode = "a" if header_written else "w"
        with open(path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(row)

    return callback


def main():
    ap = ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset class expression, e.g. 'SineClustersPre()'.")
    ap.add_argument("--size", type=int, required=True,
                    help="Ensemble size (number of detector slots).")
    ap.add_argument("--timeout", type=int, default=24 * 60 * 60,
                    help="Total optimization wall-clock budget in seconds "
                         "(default 86400 = 24h).")
    ap.add_argument("--per-trial-timeout", type=int, default=1800,
                    help="Per-trial timeout in seconds (default 1800).")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="Optional cap on the number of trials.")
    ap.add_argument("--tolerance", type=int, default=100,
                    help="Tolerance window in samples for TP matching.")
    ap.add_argument("--seed", type=int, default=1337,
                    help="Base seed forwarded to detectors / sampler.")
    ap.add_argument("--output-dir", default="synthetic_optuna_results",
                    help="Directory for CSV result files.")
    ap.add_argument("--storage", default=None,
                    help="Optional Optuna storage URL (e.g. sqlite:///foo.db). "
                         "Defaults to per-(dataset,size) sqlite file under output-dir.")
    ap.add_argument("--study-name", default=None,
                    help="Optuna study name (default auto: synthF1_<ds>_N<size>).")
    args = ap.parse_args()

    dataset_label = args.dataset.split("(", 1)[0]
    os.makedirs(args.output_dir, exist_ok=True)
    results_csv = os.path.join(
        args.output_dir,
        f"synthF1_{dataset_label}_N{args.size}.csv")

    study_name = args.study_name or f"synthF1_{dataset_label}_N{args.size}"
    storage = args.storage or (
        f"sqlite:///{os.path.join(args.output_dir, study_name + '.db')}")

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    n_existing = len(study.trials)
    print("=" * 80)
    print(f"Synthetic F1 Optuna Optimization")
    print("=" * 80)
    print(f"  Dataset           : {args.dataset}")
    print(f"  Ensemble size N   : {args.size}")
    print(f"  Tolerance         : {args.tolerance}")
    print(f"  Timeout (wall)    : {args.timeout}s")
    print(f"  Per-trial timeout : {args.per_trial_timeout}s")
    print(f"  Study             : {study_name}")
    print(f"  Storage           : {storage}")
    print(f"  Output CSV        : {results_csv}")
    print(f"  Existing trials   : {n_existing}")
    print(f"  Started           : {datetime.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80, flush=True)

    objective = make_objective(
        dataset_expr=args.dataset,
        size=args.size,
        tolerance=args.tolerance,
        seed=args.seed,
        per_trial_timeout=args.per_trial_timeout,
    )

    callback = _csv_writer(results_csv)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[callback],
        gc_after_trial=True,
    )

    print("=" * 80)
    print(f"Optimization finished. Trials run: {len(study.trials)}")
    try:
        best = study.best_trial
        print(f"Best trial #{best.number}: F1={best.value:.4f}")
        print(f"  user_attrs: {best.user_attrs}")
        print(f"  params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
    except ValueError:
        print("No completed trials.")
    print("=" * 80)


if __name__ == "__main__":
    main()
