"""
Optuna optimization of MOPEDDS ensemble F1 across MULTIPLE synthetic streams,
with a HELD-OUT evaluation subset.

For a fixed (stream_generator, ensemble_size), Optuna jointly optimizes the
same search space as ``synthetic_f1_optimize_optuna.py`` (MOPEDDS-level
params, per-slot detector class, per-slot detector hyperparameters), but the
objective is computed across ``--n-streams`` distinct streams produced by the
same synthetic generator with different fixed seeds AND different
per-stream drift frequencies.

The N streams are split into TWO disjoint subsets:

* TRAIN streams  -- visible to the optimizer; the objective is the macro F1
                    (mean of per-stream F1) over these streams.
* EVAL streams   -- held out; the optimizer never sees these. After the
                    optimization run finishes, the best trial is re-run on
                    BOTH subsets and a `*_eval.csv` row is written so the
                    plotting notebook can compare train-vs-eval generalization.

The set of stream seeds + drift frequencies is FIXED across all trials and
all workers, so every (size, trial) is evaluated on the exact same N
streams. The same train/eval index split MUST be reused across all
``--size`` runs and generators for results to be comparable.

Per-stream tolerance for TP-matching is configurable via ``--tolerances``
and broadcasts the same way as ``--drift-frequencies``. If omitted, each
tolerance defaults to ``max(1, drift_frequency // 50)`` (~2% of the
inter-drift gap), which keeps the matching window proportional to each
stream's drift density so train and eval F1 are directly comparable.

Macro F1 in [0, 1] is bounded the same way on every subset regardless of
the per-stream drift density, so unlike MTR it does not suffer from a
subset-dependent ceiling.

Supported generators:
    SineClusters(drift_frequency, stream_length, seed)
    WaveformDrift2(drift_frequency, stream_length, seed)

Usage:
    python optimization/synthetic_f1_multistream_optimize_optuna.py \\
        --generator SineClusters --size 8 --n-streams 10 \\
        --drift-frequencies 200,400,500,750,1000,1250,1500,2000,2500,3000 \\
        --stream-length 10000 --timeout 86400
"""

from __future__ import annotations

import os
import sys
import csv
import math
import signal
import logging
import warnings
import datetime
import multiprocessing as mp
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

warnings.filterwarnings("ignore")

# Make repository importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the synthetic evaluation primitives.
from main_synthetic import (  # noqa: E402
    get_detector_class,
    apply_suppression,
    evaluate_detections,
    run_ensemble,
)

# Lazy import of the generator classes (kept inside the GENERATORS map so the
# top of this file stays decoupled from dataset code paths during --help).
from datasets.sineclusters import SineClusters  # noqa: E402

try:
    from datasets.waveform import WaveformDrift2  # noqa: E402
except Exception:  # pragma: no cover - waveform module optional in some checkouts
    WaveformDrift2 = None  # type: ignore[assignment]

GENERATORS = {"SineClusters": SineClusters}
if WaveformDrift2 is not None:
    GENERATORS["WaveformDrift2"] = WaveformDrift2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate detector pool (mirror single-stream F1 optimizer)
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


def build_stream(generator_name: str, drift_frequency: int,
                 stream_length: int, seed: int):
    """Instantiate a fresh synthetic stream with the given seed."""
    cls = GENERATORS[generator_name]
    return cls(drift_frequency=drift_frequency,
               stream_length=stream_length,
               seed=seed)


def _suggest_detector_params(trial: optuna.Trial, prefix: str, kind: str) -> dict:
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
# Per-stream metric helper
# ---------------------------------------------------------------------------

def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def _run_one_stream(*, generator_name: str,
                    drift_frequency: int,
                    stream_length: int,
                    stream_seed: int,
                    tolerance: int,
                    slot_specs,
                    detector_seed_base: int,
                    s_idx: int,
                    detector_criterion: str,
                    ensemble_criterion: str,
                    decision_window: int,
                    suppression_window: int,
                    recent_samples_size: int):
    """Build a fresh stream + fresh detectors, run the ensemble, evaluate
    against the stream's known drifts. Returns
    ``(tp, fp, fn, mean_delay, f1, precision, recall, n_known)``."""
    stream = build_stream(generator_name, drift_frequency,
                          stream_length, stream_seed)
    known = list(stream.drifts)

    detectors = []
    names = []
    for i, (kind, params) in enumerate(slot_specs):
        det = _instantiate(
            kind, params,
            seed=detector_seed_base + i + 1000 * s_idx,
            recent_samples_size=recent_samples_size)
        detectors.append(det)
        names.append(f"[{i:03d}]{kind}")

    _, raw = run_ensemble(
        stream, detectors, names,
        detector_criterion=detector_criterion,
        ensemble_criterion=ensemble_criterion,
        decision_window=decision_window,
    )
    dets = apply_suppression(raw, suppression_window)
    tp, fp, fn, mean_delay = evaluate_detections(dets, known, tolerance)
    f1 = _f1_from_counts(tp, fp, fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return tp, fp, fn, mean_delay, f1, precision, recall, len(known)


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(*, generators: List[str],
                   drift_frequencies: List[int],
                   stream_length: int,
                   stream_seeds: List[int],
                   tolerances: List[int],
                   train_indices: List[int],
                   size: int,
                   detector_seed: int,
                   per_trial_timeout: int,
                   objective_metric: str,
                   pinned_globals: Dict[str, object] = None):
    assert (len(generators) == len(drift_frequencies)
            == len(stream_seeds) == len(tolerances))
    # Probe each TRAIN stream once to count its known drifts (cheap & lets
    # us validate the split).
    train_known_counts = []
    for i in train_indices:
        probe = build_stream(generators[i], drift_frequencies[i],
                             stream_length, stream_seeds[i])
        if not hasattr(probe, "drifts"):
            raise ValueError(f"Generator '{generators[i]}' has no `drifts` "
                             f"attribute on its produced stream.")
        train_known_counts.append(len(list(probe.drifts)))

    pinned_globals = dict(pinned_globals or {})
    train_tolerances = [tolerances[i] for i in train_indices]
    suppression_max = max(0, min(train_tolerances)) if train_tolerances else 0
    # Clip pinned suppression_window if it exceeds the allowed range. This
    # makes the ablation script's life easier (it can pass a fixed value
    # without knowing the per-stream tolerances).
    if "suppression_window" in pinned_globals:
        pv = int(pinned_globals["suppression_window"])
        if pv < 0 or pv > suppression_max:
            raise ValueError(
                f"--pin-globals suppression_window={pv} out of valid range "
                f"[0,{suppression_max}] given train tolerances {train_tolerances}.")

    logger.info(
        f"Generators(train)={[generators[i] for i in train_indices]} size={size} "
        f"train_indices={train_indices} "
        f"train_drift_frequencies={[drift_frequencies[i] for i in train_indices]} "
        f"train_tolerances={train_tolerances} "
        f"train_known_drifts_per_stream={train_known_counts} "
        f"suppression_max={suppression_max}")

    def objective(trial: optuna.Trial) -> float:
        # MOPEDDS-level params. Each global is either sampled by Optuna or
        # pinned via --pin-globals (and recorded as a user_attr so trials
        # remain comparable across the search history).
        def _maybe_pin_categorical(key: str, choices):
            if key in pinned_globals:
                trial.set_user_attr(f"pinned_{key}", pinned_globals[key])
                return pinned_globals[key]
            return trial.suggest_categorical(key, choices)

        def _maybe_pin_int(key: str, low: int, high: int):
            if key in pinned_globals:
                trial.set_user_attr(f"pinned_{key}", pinned_globals[key])
                return int(pinned_globals[key])
            return trial.suggest_int(key, low, high)

        det_crit = _maybe_pin_categorical(
            "detector_decision_criteria", ["any", "majority", "all"])
        ens_crit = _maybe_pin_categorical(
            "ensemble_decision_criteria", ["any", "majority", "all"])
        decision_window = _maybe_pin_int("decision_window", 1, 100)
        if suppression_max > 0:
            suppression_window = _maybe_pin_int(
                "suppression_window", 0, suppression_max)
        else:
            suppression_window = 0
        recent_samples_size = _maybe_pin_int("recent_samples_size", 50, 5000)

        # Slot composition + per-slot params
        slot_specs = []
        for i in range(size):
            kind = trial.suggest_categorical(f"slot{i}_type", CANDIDATES)
            params = _suggest_detector_params(trial, f"slot{i}_{kind}_", kind)
            slot_specs.append((kind, params))

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(per_trial_timeout)

        try:
            tp_total = 0
            fp_total = 0
            fn_total = 0
            delay_sum = 0.0
            delay_n = 0
            per_stream_f1: List[float] = []
            per_stream_precision: List[float] = []
            per_stream_recall: List[float] = []
            per_stream_tp: List[int] = []
            per_stream_fp: List[int] = []
            per_stream_fn: List[int] = []
            per_stream_n_drifts: List[int] = []
            per_stream_mean_delay: List[float] = []

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
                    detector_criterion=det_crit,
                    ensemble_criterion=ens_crit,
                    decision_window=decision_window,
                    suppression_window=suppression_window,
                    recent_samples_size=recent_samples_size,
                )
                tp_total += int(tp)
                fp_total += int(fp)
                fn_total += int(fn)
                if mean_delay == mean_delay:  # not NaN
                    delay_sum += float(mean_delay) * int(tp)
                    delay_n += int(tp)
                per_stream_f1.append(float(f1))
                per_stream_precision.append(float(prec))
                per_stream_recall.append(float(rec))
                per_stream_tp.append(int(tp))
                per_stream_fp.append(int(fp))
                per_stream_fn.append(int(fn))
                per_stream_n_drifts.append(int(n_known))
                per_stream_mean_delay.append(
                    float(mean_delay) if mean_delay == mean_delay else float("nan"))

            micro_f1 = _f1_from_counts(tp_total, fp_total, fn_total)
            macro_f1 = (sum(per_stream_f1) / len(per_stream_f1)) if per_stream_f1 else 0.0
            precision_micro = (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
            recall_micro = (tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0
            mean_delay_micro = (delay_sum / delay_n) if delay_n > 0 else float("nan")

            trial.set_user_attr("micro_f1", float(micro_f1))
            trial.set_user_attr("macro_f1", float(macro_f1))
            trial.set_user_attr("tp_total", int(tp_total))
            trial.set_user_attr("fp_total", int(fp_total))
            trial.set_user_attr("fn_total", int(fn_total))
            trial.set_user_attr("precision_micro", float(precision_micro))
            trial.set_user_attr("recall_micro", float(recall_micro))
            trial.set_user_attr("mean_delay_micro", float(mean_delay_micro))
            trial.set_user_attr("per_stream_f1", per_stream_f1)
            trial.set_user_attr("per_stream_precision", per_stream_precision)
            trial.set_user_attr("per_stream_recall", per_stream_recall)
            trial.set_user_attr("per_stream_tp", per_stream_tp)
            trial.set_user_attr("per_stream_fp", per_stream_fp)
            trial.set_user_attr("per_stream_fn", per_stream_fn)
            trial.set_user_attr("per_stream_n_drifts", per_stream_n_drifts)
            trial.set_user_attr("per_stream_mean_delay", per_stream_mean_delay)
            trial.set_user_attr("train_indices", list(train_indices))
            trial.set_user_attr("per_stream_tolerance",
                                [tolerances[i] for i in train_indices])
            trial.set_user_attr("n_train_streams", len(train_indices))

            logger.info(
                f"Trial {trial.number}: macroF1={macro_f1:.4f} "
                f"microF1={micro_f1:.4f} P={precision_micro:.3f} "
                f"R={recall_micro:.3f} TP={tp_total} FP={fp_total} FN={fn_total} "
                f"per_stream_f1={['%.3f' % v for v in per_stream_f1]}")

            return float(macro_f1 if objective_metric == "macro" else micro_f1)

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
# CSV writer
# ---------------------------------------------------------------------------

_FIXED_COLS = [
    "trial_id", "macro_f1", "micro_f1",
    "tp_total", "fp_total", "fn_total",
    "precision_micro", "recall_micro", "mean_delay_micro",
    "per_stream_f1", "per_stream_precision", "per_stream_recall",
    "per_stream_tp", "per_stream_fp", "per_stream_fn",
    "per_stream_n_drifts", "per_stream_mean_delay",
    "train_indices", "per_stream_tolerance", "n_train_streams",
    "error",
    "detector_decision_criteria", "ensemble_decision_criteria",
    "decision_window", "suppression_window", "recent_samples_size",
    "slot0_type",
]

# Pre-define all possible detector parameter columns to avoid race conditions in parallel runs
_DETECTOR_PARAM_COLS = {
    "BNDM": ["slot0_BNDM_n_samples", "slot0_BNDM_const", "slot0_BNDM_threshold", "slot0_BNDM_max_depth"],
    "CSDDM": ["slot0_CSDDM_n_samples", "slot0_CSDDM_feature_proportion", "slot0_CSDDM_n_clusters", "slot0_CSDDM_confidence"],
    "D3": ["slot0_D3_n_reference_samples", "slot0_D3_recent_samples_proportion", "slot0_D3_threshold"],
    "IBDD": ["slot0_IBDD_n_samples", "slot0_IBDD_n_consecutive_deviations", "slot0_IBDD_n_permutations", "slot0_IBDD_update_interval"],
    "OCDD": ["slot0_OCDD_n_samples", "slot0_OCDD_threshold"],
    "SPLL": ["slot0_SPLL_n_samples", "slot0_SPLL_n_clusters", "slot0_SPLL_threshold"],
    "UDetect": ["slot0_UDetect_n_windows", "slot0_UDetect_n_samples", "slot0_UDetect_disjoint_training_windows"],
}

# Flatten all detector param columns
_ALL_PARAM_COLS = []
for cols in _DETECTOR_PARAM_COLS.values():
    _ALL_PARAM_COLS.extend(cols)

# Complete fieldnames for CSV
_ALL_FIELDNAMES = _FIXED_COLS + _ALL_PARAM_COLS


def _csv_writer(path: str):
    """Return a callback that appends each completed trial to ``path``.
    
    Uses pre-defined fieldnames to avoid race conditions in parallel runs.
    """
    header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def callback(study, trial):
        nonlocal header_written
        if trial.state != TrialState.COMPLETE:
            return
        row = {
            "trial_id": trial.number,
            "macro_f1": trial.user_attrs.get("macro_f1",
                                             trial.value if trial.value is not None else ""),
            "micro_f1": trial.user_attrs.get("micro_f1", ""),
            "tp_total": trial.user_attrs.get("tp_total", ""),
            "fp_total": trial.user_attrs.get("fp_total", ""),
            "fn_total": trial.user_attrs.get("fn_total", ""),
            "precision_micro": trial.user_attrs.get("precision_micro", ""),
            "recall_micro": trial.user_attrs.get("recall_micro", ""),
            "mean_delay_micro": trial.user_attrs.get("mean_delay_micro", ""),
            "per_stream_f1": trial.user_attrs.get("per_stream_f1", ""),
            "per_stream_precision": trial.user_attrs.get("per_stream_precision", ""),
            "per_stream_recall": trial.user_attrs.get("per_stream_recall", ""),
            "per_stream_tp": trial.user_attrs.get("per_stream_tp", ""),
            "per_stream_fp": trial.user_attrs.get("per_stream_fp", ""),
            "per_stream_fn": trial.user_attrs.get("per_stream_fn", ""),
            "per_stream_n_drifts": trial.user_attrs.get("per_stream_n_drifts", ""),
            "per_stream_mean_delay": trial.user_attrs.get("per_stream_mean_delay", ""),
            "train_indices": trial.user_attrs.get("train_indices", ""),
            "per_stream_tolerance": trial.user_attrs.get("per_stream_tolerance", ""),
            "n_train_streams": trial.user_attrs.get("n_train_streams", ""),
            "error": trial.user_attrs.get("error", ""),
        }
        row.update(trial.params)
        # Ensure all pre-defined columns exist in row
        for k in _ALL_FIELDNAMES:
            row.setdefault(k, "")
        mode = "a" if header_written else "w"
        with open(path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_ALL_FIELDNAMES,
                                    extrasaction="ignore")
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(row)

    return callback


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _resolve_stream_seeds(arg_seeds: str, n_streams: int,
                          base_stream_seed: int) -> List[int]:
    if arg_seeds:
        seeds = [int(s.strip()) for s in arg_seeds.split(",") if s.strip()]
        if len(seeds) != n_streams:
            raise ValueError(
                f"--stream-seeds has {len(seeds)} values but "
                f"--n-streams is {n_streams}.")
        return seeds
    return [base_stream_seed + i for i in range(n_streams)]


def _resolve_list(csv_str: str, n_streams: int, name: str) -> List[int]:
    vals = [int(s.strip()) for s in csv_str.split(",") if s.strip()]
    if len(vals) == 1:
        return [vals[0]] * n_streams
    if len(vals) != n_streams:
        raise ValueError(
            f"{name} has {len(vals)} values but --n-streams is {n_streams}.")
    return vals


def _resolve_generators(arg_generators: str, legacy_generator: str,
                        n_streams: int) -> List[str]:
    """Resolve the per-stream list of generator names. ``--generators`` (a
    comma-separated list of length 1 or n_streams) takes precedence over the
    legacy single-value ``--generator`` flag."""
    raw = arg_generators if arg_generators else legacy_generator
    if not raw:
        raise ValueError("Must pass --generators (preferred) or --generator.")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--generators / --generator produced an empty list.")
    if len(parts) == 1:
        gens = parts * n_streams
    elif len(parts) == n_streams:
        gens = parts
    else:
        raise ValueError(
            f"--generators has {len(parts)} entries but --n-streams is "
            f"{n_streams}; must be length 1 (broadcast) or --n-streams.")
    bad = sorted({g for g in gens if g not in GENERATORS})
    if bad:
        raise ValueError(f"Unknown generator name(s): {bad}. "
                         f"Allowed: {sorted(GENERATORS.keys())}")
    return gens


_PINNABLE_GLOBAL_KEYS = (
    "detector_decision_criteria",
    "ensemble_decision_criteria",
    "decision_window",
    "suppression_window",
    "recent_samples_size",
)
_PINNABLE_GLOBAL_CATEGORICAL = {
    "detector_decision_criteria": ("any", "majority", "all"),
    "ensemble_decision_criteria": ("any", "majority", "all"),
}


def _parse_pin_globals(spec: str) -> Dict[str, object]:
    """Parse a 'key1=val1,key2=val2' string into a validated pin dict. Empty
    or None input returns an empty dict. Integer-valued keys are coerced to
    int; categorical keys are validated against the allowed value sets."""
    if not spec:
        return {}
    out: Dict[str, object] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--pin-globals entry '{part}' missing '='.")
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key not in _PINNABLE_GLOBAL_KEYS:
            raise ValueError(
                f"--pin-globals key '{key}' not pinnable. Allowed: "
                f"{list(_PINNABLE_GLOBAL_KEYS)}")
        if key in _PINNABLE_GLOBAL_CATEGORICAL:
            choices = _PINNABLE_GLOBAL_CATEGORICAL[key]
            if val not in choices:
                raise ValueError(
                    f"--pin-globals {key}={val} not in {list(choices)}.")
            out[key] = val
        else:
            try:
                out[key] = int(val)
            except ValueError as e:
                raise ValueError(
                    f"--pin-globals {key}={val} expected int.") from e
    return out


def _resolve_study_tag(arg_tag: str, generators: List[str]) -> str:
    """Tag used in CSV/study names. If the user passes one, use it. Otherwise
    derive a stable name from the per-stream generator list."""
    if arg_tag:
        return arg_tag
    uniq = sorted(set(generators))
    if len(uniq) == 1:
        return uniq[0]
    return "Mix_" + "+".join(uniq)


def _default_tolerances(drift_frequencies: List[int]) -> List[int]:
    """Default per-stream TP-matching tolerance: ~2% of the inter-drift gap,
    with a floor of 1 sample. Keeps the matching window proportional across
    streams so train and eval F1 are directly comparable."""
    return [max(1, df // 50) for df in drift_frequencies]


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

def _reconstruct_slot_specs(params: dict, size: int):
    """Rebuild the per-slot detector kind + hyperparameter dicts from a flat
    ``trial.params`` dictionary."""
    slot_specs = []
    for i in range(size):
        kind = params[f"slot{i}_type"]
        prefix = f"slot{i}_{kind}_"
        sub = {k[len(prefix):]: v for k, v in params.items()
               if k.startswith(prefix)}
        slot_specs.append((kind, sub))
    return slot_specs


def _evaluate_subset(*, generators: List[str],
                     drift_frequencies: List[int],
                     stream_length: int,
                     stream_seeds: List[int],
                     tolerances: List[int],
                     indices: List[int],
                     slot_specs,
                     detector_seed: int,
                     detector_criterion: str,
                     ensemble_criterion: str,
                     decision_window: int,
                     suppression_window: int,
                     recent_samples_size: int):
    per_f1, per_p, per_r = [], [], []
    per_tp, per_fp, per_fn = [], [], []
    per_n_drifts, per_delay = [], []
    tp_total = fp_total = fn_total = 0
    for s_idx in indices:
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
        per_f1.append(float(f1))
        per_p.append(float(prec))
        per_r.append(float(rec))
        per_tp.append(int(tp))
        per_fp.append(int(fp))
        per_fn.append(int(fn))
        per_n_drifts.append(int(n_known))
        per_delay.append(float(mean_delay) if mean_delay == mean_delay else float("nan"))
        tp_total += int(tp)
        fp_total += int(fp)
        fn_total += int(fn)
    macro_f1 = (sum(per_f1) / len(per_f1)) if per_f1 else 0.0
    micro_f1 = _f1_from_counts(tp_total, fp_total, fn_total)
    return {
        "macro_f1": macro_f1, "micro_f1": micro_f1,
        "tp_total": tp_total, "fp_total": fp_total, "fn_total": fn_total,
        "per_f1": per_f1, "per_precision": per_p, "per_recall": per_r,
        "per_tp": per_tp, "per_fp": per_fp, "per_fn": per_fn,
        "per_n_drifts": per_n_drifts, "per_mean_delay": per_delay,
    }


def _evaluate_best_on_held_out(*, best, generators: List[str],
                               study_tag: str,
                               drift_frequencies: List[int],
                               stream_length: int,
                               stream_seeds: List[int],
                               tolerances: List[int],
                               eval_indices: List[int],
                               train_indices: List[int],
                               size: int,
                               detector_seed: int,
                               eval_csv: str) -> None:
    params = dict(best.params)
    slot_specs = _reconstruct_slot_specs(params, size)
    det_crit = params["detector_decision_criteria"]
    ens_crit = params["ensemble_decision_criteria"]
    decision_window = int(params["decision_window"])
    suppression_window = int(params.get("suppression_window", 0))
    recent_samples_size = int(params["recent_samples_size"])

    kwargs = dict(generators=generators,
                  drift_frequencies=drift_frequencies,
                  stream_length=stream_length,
                  stream_seeds=stream_seeds,
                  tolerances=tolerances,
                  slot_specs=slot_specs,
                  detector_seed=detector_seed,
                  detector_criterion=det_crit,
                  ensemble_criterion=ens_crit,
                  decision_window=decision_window,
                  suppression_window=suppression_window,
                  recent_samples_size=recent_samples_size)
    train_m = _evaluate_subset(indices=train_indices, **kwargs)
    eval_m = _evaluate_subset(indices=eval_indices, **kwargs)

    print("-" * 80)
    print(f"Held-out evaluation of best trial #{best.number}:")
    print(f"  study tag    = {study_tag}")
    print(f"  train indices = {train_indices}")
    print(f"    generators  = {[generators[i] for i in train_indices]}")
    print(f"    drift freqs = {[drift_frequencies[i] for i in train_indices]}")
    print(f"    tolerances  = {[tolerances[i] for i in train_indices]}")
    print(f"    macroF1     = {train_m['macro_f1']:.4f}   "
          f"microF1 = {train_m['micro_f1']:.4f}")
    print(f"  eval  indices = {eval_indices}")
    print(f"    generators  = {[generators[i] for i in eval_indices]}")
    print(f"    drift freqs = {[drift_frequencies[i] for i in eval_indices]}")
    print(f"    tolerances  = {[tolerances[i] for i in eval_indices]}")
    print(f"    macroF1     = {eval_m['macro_f1']:.4f}   "
          f"microF1 = {eval_m['micro_f1']:.4f}")
    print(f"  gap (train-eval) macroF1 = "
          f"{train_m['macro_f1'] - eval_m['macro_f1']:+.4f}")
    print("-" * 80, flush=True)

    row = {
        "trial_id": best.number,
        "generator": study_tag,
        "study_tag": study_tag,
        "size": size,
        "n_streams": len(stream_seeds),
        "train_indices": list(train_indices),
        "eval_indices": list(eval_indices),
        "train_generators": [generators[i] for i in train_indices],
        "eval_generators": [generators[i] for i in eval_indices],
        "train_drift_frequencies": [drift_frequencies[i] for i in train_indices],
        "eval_drift_frequencies": [drift_frequencies[i] for i in eval_indices],
        "train_tolerances": [tolerances[i] for i in train_indices],
        "eval_tolerances": [tolerances[i] for i in eval_indices],
        "train_macro_f1": train_m["macro_f1"],
        "eval_macro_f1": eval_m["macro_f1"],
        "train_micro_f1": train_m["micro_f1"],
        "eval_micro_f1": eval_m["micro_f1"],
        "train_tp_total": train_m["tp_total"],
        "train_fp_total": train_m["fp_total"],
        "train_fn_total": train_m["fn_total"],
        "eval_tp_total": eval_m["tp_total"],
        "eval_fp_total": eval_m["fp_total"],
        "eval_fn_total": eval_m["fn_total"],
        "train_per_f1": train_m["per_f1"],
        "train_per_precision": train_m["per_precision"],
        "train_per_recall": train_m["per_recall"],
        "train_per_tp": train_m["per_tp"],
        "train_per_fp": train_m["per_fp"],
        "train_per_fn": train_m["per_fn"],
        "train_per_n_drifts": train_m["per_n_drifts"],
        "train_per_mean_delay": train_m["per_mean_delay"],
        "eval_per_f1": eval_m["per_f1"],
        "eval_per_precision": eval_m["per_precision"],
        "eval_per_recall": eval_m["per_recall"],
        "eval_per_tp": eval_m["per_tp"],
        "eval_per_fp": eval_m["per_fp"],
        "eval_per_fn": eval_m["per_fn"],
        "eval_per_n_drifts": eval_m["per_n_drifts"],
        "eval_per_mean_delay": eval_m["per_mean_delay"],
    }
    # Persist params for traceability / re-run.
    for k, v in params.items():
        row[f"param_{k}"] = v

    # Use pre-defined fieldnames for eval CSV to avoid race conditions
    eval_fieldnames = [
        "kind", "params", "detector_decision_criteria", "ensemble_decision_criteria",
        "decision_window", "suppression_window", "recent_samples_size",
        "eval_macro_f1", "eval_per_stream_f1",
        "train_macro_f1", "train_per_stream_f1",
        "train_micro_f1", "eval_micro_f1",
        "train_tp_total", "train_fp_total", "train_fn_total",
        "eval_tp_total", "eval_fp_total", "eval_fn_total",
        "train_per_precision", "train_per_recall", "train_per_tp", "train_per_fp", "train_per_fn",
        "train_per_n_drifts", "train_per_mean_delay",
        "eval_per_precision", "eval_per_recall", "eval_per_tp", "eval_per_fp", "eval_per_fn",
        "eval_per_n_drifts", "eval_per_mean_delay",
    ]
    # Add param_ columns
    for k in params.keys():
        if f"param_{k}" not in eval_fieldnames:
            eval_fieldnames.append(f"param_{k}")
    
    write_header = not os.path.exists(eval_csv) or os.path.getsize(eval_csv) == 0
    with open(eval_csv, "a" if not write_header else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for k in eval_fieldnames:
            row.setdefault(k, "")
        writer.writerow(row)
    print(f"Wrote held-out eval row to: {eval_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = ArgumentParser()
    ap.add_argument("--generator", default=None, choices=list(GENERATORS.keys()),
                    help="DEPRECATED: single stream generator class. Use "
                         "--generators for mixed-generator runs. If given "
                         "alone, it is broadcast to all N streams.")
    ap.add_argument("--generators", default=None,
                    help="Comma-separated per-stream generator names. Must "
                         "have length 1 (broadcast) or --n-streams. Overrides "
                         "--generator. Allowed values: "
                         + ", ".join(sorted(GENERATORS.keys())) + ".")
    ap.add_argument("--study-tag", default=None,
                    help="Identifier used in CSV / Optuna study names. "
                         "Defaults to the single generator name when all "
                         "streams share a generator, otherwise 'Mix_' joined "
                         "by '+' of the sorted unique generator names.")
    ap.add_argument("--pin-globals", default=None,
                    help="Comma-separated 'key=value' overrides that pin one "
                         "or more MOPEDDS-level globals during the Optuna "
                         "search (the corresponding dimension is removed from "
                         "the search space). Pinnable keys: "
                         + ", ".join(_PINNABLE_GLOBAL_KEYS) + ".")
    ap.add_argument("--size", type=int, required=True,
                    help="Ensemble size (number of detector slots).")
    ap.add_argument("--n-streams", type=int, default=10,
                    help="Number of distinct streams to evaluate per trial "
                         "(default 10).")
    ap.add_argument("--base-stream-seed", type=int, default=42,
                    help="Base seed; the N stream seeds are base+0..base+N-1 "
                         "unless --stream-seeds is given (default 42).")
    ap.add_argument("--stream-seeds", default=None,
                    help="Optional comma-separated explicit list of stream "
                         "seeds. Must have length --n-streams.")
    ap.add_argument("--drift-frequencies",
                    default="200,400,500,750,1000,1250,1500,2000,2500,3000",
                    help="Comma-separated per-stream drift frequencies. If a "
                         "single value is given, it is broadcast to all N "
                         "streams. Must have length 1 or --n-streams.")
    ap.add_argument("--stream-length", type=int, default=10000,
                    help="Stream length in samples (default 10000), used for "
                         "every stream.")
    ap.add_argument("--tolerances", default=None,
                    help="Comma-separated per-stream TP-matching tolerance "
                         "windows. If omitted each defaults to "
                         "max(1, drift_frequency // 50) (~2%% of the "
                         "inter-drift gap), which keeps the matching window "
                         "proportional across streams. If a single value is "
                         "given it is broadcast to all N streams.")
    ap.add_argument("--objective", choices=["macro", "micro"], default="macro",
                    help="Aggregation across train streams used as the Optuna "
                         "objective. 'macro' = mean of per-stream F1 (default); "
                         "'micro' = F1 computed on summed TP/FP/FN.")
    ap.add_argument("--timeout", type=int, default=24 * 60 * 60,
                    help="Total optimization wall-clock budget in seconds.")
    ap.add_argument("--per-trial-timeout", type=int, default=3600,
                    help="Per-trial timeout in seconds (default 3600). Trials "
                         "evaluate N_train streams so this should be larger "
                         "than the single-stream optimizer.")
    ap.add_argument("--n-trials", type=int, default=1000,
                    help="Global cap on SUCCESSFUL trials across workers.")
    ap.add_argument("--eval-stream-indices", default="1,4,8",
                    help="Comma-separated indices (into the full stream list) "
                         "used as the HELD-OUT evaluation subset. The remaining "
                         "indices form the training subset that the optimizer "
                         "actually sees. Default '1,4,8' stratifies the split "
                         "by drift frequency with the default "
                         "--drift-frequencies "
                         "[200,400,500,750,1000,1250,1500,2000,2500,3000]: "
                         "eval df {400, 1000, 2500} (mean 1300) vs train df "
                         "{200, 500, 750, 1250, 1500, 2000, 3000} (mean 1314). "
                         "The same split MUST be reused across all --size runs "
                         "and generators for the results to be comparable.")
    ap.add_argument("--seed", type=int, default=1337,
                    help="Base seed for Optuna sampler / detectors.")
    ap.add_argument("--output-dir", default="synthetic_multistream_results",
                    help="Directory for CSV result files / sqlite DB.")
    ap.add_argument("--storage", default=None,
                    help="Optional Optuna storage URL.")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--n-workers", type=int, default=1)
    args = ap.parse_args()

    stream_seeds = _resolve_stream_seeds(args.stream_seeds, args.n_streams,
                                         args.base_stream_seed)
    drift_frequencies = _resolve_list(
        args.drift_frequencies, args.n_streams, name="--drift-frequencies")
    generators_list = _resolve_generators(args.generators, args.generator,
                                          args.n_streams)
    study_tag = _resolve_study_tag(args.study_tag, generators_list)
    pinned_globals = _parse_pin_globals(args.pin_globals)
    if args.tolerances is None:
        tolerances = _default_tolerances(drift_frequencies)
    else:
        tolerances = _resolve_list(args.tolerances, args.n_streams,
                                   name="--tolerances")

    eval_indices = sorted({int(s.strip())
                           for s in args.eval_stream_indices.split(",")
                           if s.strip() != ""})
    for idx in eval_indices:
        if idx < 0 or idx >= args.n_streams:
            raise ValueError(
                f"--eval-stream-indices contains out-of-range index {idx} "
                f"(must be in [0, {args.n_streams - 1}]).")
    train_indices = [i for i in range(args.n_streams) if i not in set(eval_indices)]
    if not train_indices:
        raise ValueError("All stream indices were assigned to evaluation; "
                         "need at least one training stream.")
    if not eval_indices:
        raise ValueError("No evaluation indices given; pass at least one via "
                         "--eval-stream-indices.")

    os.makedirs(args.output_dir, exist_ok=True)
    study_name = args.study_name or (
        f"synthF1ms_{study_tag}_N{args.size}_S{args.n_streams}")
    storage = args.storage or (
        f"sqlite:///{os.path.join(args.output_dir, study_name + '.db')}")

    storage_obj = _build_storage(storage)
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_obj,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    n_existing = len(study.trials)
    print("=" * 80)
    print("Synthetic Multi-Stream F1 Optuna Optimization")
    print("=" * 80)
    print(f"  Study tag         : {study_tag}")
    print(f"  Generators (all)  : {generators_list}")
    if pinned_globals:
        print(f"  Pinned globals    : {pinned_globals}")
    print(f"  Ensemble size N   : {args.size}")
    print(f"  Streams per trial : {args.n_streams}")
    print(f"  Stream seeds      : {stream_seeds}")
    print(f"  Drift frequencies : {drift_frequencies}")
    print(f"  Tolerances        : {tolerances}")
    print(f"  Stream length     : {args.stream_length}")
    print(f"  Train indices     : {train_indices}")
    print(f"    generators      : {[generators_list[i] for i in train_indices]}")
    print(f"    drift freqs     : {[drift_frequencies[i] for i in train_indices]}")
    print(f"    tolerances      : {[tolerances[i] for i in train_indices]}")
    print(f"    seeds           : {[stream_seeds[i] for i in train_indices]}")
    print(f"  Eval indices      : {eval_indices}")
    print(f"    generators      : {[generators_list[i] for i in eval_indices]}")
    print(f"    drift freqs     : {[drift_frequencies[i] for i in eval_indices]}")
    print(f"    tolerances      : {[tolerances[i] for i in eval_indices]}")
    print(f"    seeds           : {[stream_seeds[i] for i in eval_indices]}")
    print(f"  Objective metric  : {args.objective}_f1")
    print(f"  Suppression max   : "
          f"{max(0, min([tolerances[i] for i in train_indices]))}  "
          f"(= min(train_tolerances))")
    print(f"  Timeout (wall)    : {args.timeout}s")
    print(f"  Per-trial timeout : {args.per_trial_timeout}s")
    print(f"  Study             : {study_name}")
    print(f"  Storage           : {storage}")
    print(f"  Workers           : {args.n_workers}")
    print(f"  Existing trials   : {n_existing}")
    print(f"  Started           : {datetime.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80, flush=True)

    n_workers = max(1, int(args.n_workers))
    if n_workers == 1:
        _run_worker(args, study_name, storage, stream_seeds,
                    drift_frequencies, tolerances, train_indices,
                    generators_list, study_tag, pinned_globals, worker_idx=0)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for w in range(n_workers):
            p = ctx.Process(
                target=_run_worker,
                args=(args, study_name, storage, stream_seeds,
                      drift_frequencies, tolerances, train_indices,
                      generators_list, study_tag, pinned_globals, w),
                name=f"optuna-worker-{w}",
            )
            p.start()
            procs.append(p)
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join()

    final_study = optuna.load_study(study_name=study_name,
                                    storage=_build_storage(storage))
    print("=" * 80)
    print(f"Optimization finished. Trials run: {len(final_study.trials)}")
    try:
        best = final_study.best_trial
        print(f"Best trial #{best.number}: train {args.objective}F1="
              f"{best.value:.4f}")
        print(f"  user_attrs: {best.user_attrs}")
        print(f"  params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        eval_csv = os.path.join(
            args.output_dir,
            f"synthF1ms_{study_tag}_N{args.size}_S{args.n_streams}_eval.csv")
        _evaluate_best_on_held_out(
            best=best,
            generators=generators_list,
            study_tag=study_tag,
            drift_frequencies=drift_frequencies,
            stream_length=args.stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            eval_indices=eval_indices,
            train_indices=train_indices,
            size=args.size,
            detector_seed=args.seed,
            eval_csv=eval_csv,
        )
    except ValueError:
        print("No completed trials.")


def _build_storage(storage_url: str):
    if storage_url.startswith("sqlite:"):
        return optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
    return storage_url


def _run_worker(args, study_name: str, storage_url: str,
                stream_seeds: List[int],
                drift_frequencies: List[int],
                tolerances: List[int],
                train_indices: List[int],
                generators: List[str],
                study_tag: str,
                pinned_globals: Dict[str, object],
                worker_idx: int):
    sampler = TPESampler(seed=args.seed + worker_idx)
    study = optuna.create_study(
        study_name=study_name,
        storage=_build_storage(storage_url),
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    suffix = "" if args.n_workers <= 1 else f"_w{worker_idx}"
    results_csv = os.path.join(
        args.output_dir,
        f"synthF1ms_{study_tag}_N{args.size}_S{args.n_streams}{suffix}.csv")

    objective = make_objective(
        generators=generators,
        drift_frequencies=drift_frequencies,
        stream_length=args.stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        train_indices=train_indices,
        size=args.size,
        detector_seed=args.seed + 1000 * worker_idx,
        per_trial_timeout=args.per_trial_timeout,
        objective_metric=args.objective,
        pinned_globals=pinned_globals,
    )
    callbacks = [_csv_writer(results_csv)]

    total_cap = args.n_trials if (args.n_trials and args.n_trials > 0) else None
    if total_cap is not None:
        def _global_cap_cb(study_, trial_, cap=total_cap):
            n_successful = sum(
                1 for t in study_.get_trials(deepcopy=False,
                                             states=(TrialState.COMPLETE,))
                if not t.user_attrs.get("error")
            )
            if n_successful >= cap:
                study_.stop()
        callbacks.append(_global_cap_cb)

    study.optimize(
        objective,
        n_trials=None,
        timeout=args.timeout,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=callbacks,
        gc_after_trial=True,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
