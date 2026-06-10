#!/usr/bin/env python3
"""
Drift Detector Benchmark - Synthetic Streams Evaluation Script

Runs an ensemble of drift detectors on a synthetic stream with known drift
points. No classifier is involved in the pipeline. For a list of ensemble
sizes (default: 2, 4, 8, 16, 32, 64, 128), the script reports:
    - Ensemble metrics:        TP, FP, FN (misses), mean detection delay
    - Per-member metrics:      same metrics, treating each detector as if it
                               had been deployed alone

The detector pool is provided via a YAML configuration file in the same
format as ``detectors/mopedds/configs/mopedds.config`` and must contain at
least ``max(ensemble_sizes)`` detector entries.

Usage:
    python main_synthetic.py <Dataset> <ConfigPath> [options]

Required arguments:
    Dataset         Dataset class name or expression
                    (e.g., 'SineClustersPre()'). Must expose
                    ``stream.drifts`` and ``stream.n_samples``.
    ConfigPath      Path to the YAML config defining the detector pool.

Options:
    --sizes A,B,C                       Comma separated ensemble sizes
                                        (default 2,4,8,16,32,64,128).
    --tolerance N                       Tolerance window (in samples) for
                                        counting a detection as a TP after a
                                        known drift point (default 1000).
    --suppression-window N              Per-source suppression window in
                                        samples used when collapsing repeated
                                        detections before TP/FP counting.
                                        Default: value from YAML, else
                                        --tolerance.
    --detector-decision-criteria X      Level-1 (per-detector, over
                                        --decision-window) criterion.
                                        any|all|majority. Default: value
                                        from YAML, else 'majority'.
    --ensemble-decision-criteria X      Level-2 (across detectors) criterion.
                                        any|all|majority. Default: value
                                        from YAML, else 'majority'.
    --decision-window N                 Level-1 sliding window length in
                                        samples. Default: value from YAML,
                                        else 1.
    --recent-samples-size N             Override recent_samples_size for all
                                        detectors.
    --seed N                            Base seed (default 1337).

CLI values override matching keys in the YAML config; if neither is set,
the hard-coded defaults above apply.

Example:
    python main_synthetic.py "SineClustersPre()" \
        detectors/mopedds/configs/mopedds.config --sizes 2,4,8 --tolerance 500
"""

import os
import sys
import ast
import math
import argparse
import importlib
import warnings
from collections import deque
from itertools import islice
from typing import List, Optional, Tuple

import yaml

warnings.filterwarnings("ignore")

from datasets import *  # noqa: F401,F403  - dataset classes used via eval()


# ---------------------------------------------------------------------------
# Argument / config parsing helpers
# ---------------------------------------------------------------------------

def parse_expression(expr_str):
    """Parse 'ClassName(arg=value, ...)' or 'ClassName' -> (class_name, obj)."""
    try:
        node = ast.parse(expr_str, mode='eval').body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            class_name = node.func.id
            obj = eval(expr_str, globals())
            return class_name, obj
        elif isinstance(node, ast.Name):
            class_name = node.id
            obj = eval(expr_str + "()", globals())
            return class_name, obj
        else:
            raise ValueError(f"Invalid expression format: {expr_str}")
    except Exception as e:
        raise ValueError(f"Error parsing expression '{expr_str}': {e}")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_detector_class(class_path: str):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_detector(detector_config: dict, seed: int,
                         recent_samples_size: int):
    class_path = detector_config['class']
    params = dict(detector_config.get('params', {}))
    params.setdefault('seed', seed)
    if recent_samples_size is not None:
        params['recent_samples_size'] = recent_samples_size
    cls = get_detector_class(class_path)
    return cls(**params), class_path.rsplit('.', 1)[-1]


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_detections(detections: List[int], known_drifts: List[int],
                        tolerance: int) -> Tuple[int, int, int, float]:
    """Greedy 1:1 matching between known drifts and detections.

    A detection ``d`` matches a known drift ``k`` iff ``k <= d <= k + tolerance``.
    Each known drift consumes at most one detection (the earliest unmatched).

    Returns:
        tp, fp, fn, mean_detection_delay (NaN if no TPs).
    """
    detections = sorted(set(int(d) for d in detections))
    known_drifts = sorted(int(k) for k in known_drifts)
    matched = [False] * len(detections)
    delays: List[int] = []
    fn = 0

    for kd in known_drifts:
        chosen = -1
        for i, d in enumerate(detections):
            if matched[i]:
                continue
            if d < kd:
                continue
            if d > kd + tolerance:
                break
            chosen = i
            break
        if chosen >= 0:
            matched[chosen] = True
            delays.append(detections[chosen] - kd)
        else:
            fn += 1

    tp = len(delays)
    fp = sum(1 for m in matched if not m)
    mean_delay = float(sum(delays) / len(delays)) if delays else float('nan')
    return tp, fp, fn, mean_delay


def apply_suppression(raw_detection_indices: List[int],
                      suppression: int) -> List[int]:
    """Collapse a sequence of detection sample indices using a suppression
    window: after a detection at index ``i``, subsequent detections within
    ``[i+1, i+suppression]`` are discarded."""
    if suppression <= 0:
        return list(raw_detection_indices)
    out: List[int] = []
    last = -math.inf
    for idx in raw_detection_indices:
        if idx - last > suppression:
            out.append(idx)
            last = idx
    return out


def ensemble_decision(raw_results: List[bool], criterion: str) -> bool:
    n = len(raw_results)
    if n == 0:
        return False
    s = sum(1 for r in raw_results if r)
    if criterion == "any":
        return s > 0
    if criterion == "all":
        return s == n
    return s >= (n + 1) // 2  # majority


# ---------------------------------------------------------------------------
# Main per-size runner
# ---------------------------------------------------------------------------

def run_ensemble(stream, detectors: List, detector_names: List[str],
                 detector_criterion: str,
                 ensemble_criterion: str,
                 decision_window: int
                 ) -> Tuple[List[List[int]], List[int]]:
    """Iterate the entire stream once, calling ``update`` on every detector.

    Each detector's raw output is smoothed by a sliding ``decision_window`` and
    a ``detector_criterion`` (level 1). The per-sample level-1 decisions are
    then combined by ``ensemble_criterion`` (level 2) to produce the ensemble
    decision.

    Returns:
        per_member_detections: list (one per detector) of sample indices where
            the detector's RAW output was True (i.e. "deployed alone" behavior).
        ensemble_detections: sample indices where the ensemble criterion was
            satisfied over the level-1 decisions.
    """
    per_member: List[List[int]] = [[] for _ in detectors]
    ensemble: List[int] = []
    histories: List[deque] = [deque(maxlen=max(1, decision_window))
                              for _ in detectors]

    for i, (x, _y) in enumerate(stream):
        level1_results: List[bool] = []
        for j, det in enumerate(detectors):
            try:
                triggered = bool(det.update(x))
            except Exception as e:
                triggered = False
                print(f"[warn] detector {detector_names[j]} raised {e!r} at sample {i}")
            if triggered:
                per_member[j].append(i)
            histories[j].append(triggered)
            level1_results.append(
                _apply_window_criterion(histories[j], detector_criterion)
            )
        if ensemble_decision(level1_results, ensemble_criterion):
            ensemble.append(i)

    return per_member, ensemble


def _apply_window_criterion(history: deque, criterion: str) -> bool:
    n = len(history)
    if n == 0:
        return False
    s = sum(1 for r in history if r)
    if criterion == "any":
        return s > 0
    if criterion == "all":
        return s == n
    return s >= (n + 1) // 2  # majority


def format_metric_row(name: str, tp: int, fp: int, fn: int, delay: float) -> str:
    delay_str = f"{delay:.2f}" if not math.isnan(delay) else "nan"
    return f"{name:<30s}  TP={tp:<4d}  FP={fp:<4d}  FN={fn:<4d}  delay={delay_str}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="Dataset class name or expression.")
    p.add_argument("config", help="Path to YAML detector pool config.")
    p.add_argument("--sizes", default="2,4,8,16,32,64,128",
                   help="Comma-separated ensemble sizes.")
    p.add_argument("--tolerance", type=int, default=1000,
                   help="Tolerance window in samples for TP matching.")
    p.add_argument("--suppression-window", type=int, default=None,
                   help="Per-source suppression window in samples for the FP/TP"
                        " counting collapse (default: value from YAML, else "
                        "--tolerance).")
    p.add_argument("--detector-decision-criteria", default=None,
                   choices=["any", "all", "majority"],
                   help="Level-1 (per-detector, over decision_window) criterion. "
                        "Default: value from YAML, else 'majority'.")
    p.add_argument("--ensemble-decision-criteria", default=None,
                   choices=["any", "all", "majority"],
                   help="Level-2 (across detectors) criterion. "
                        "Default: value from YAML, else 'majority'.")
    p.add_argument("--decision-window", type=int, default=None,
                   help="Level-1 sliding window length in samples. "
                        "Default: value from YAML, else 1.")
    p.add_argument("--recent-samples-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args(argv)


def _resolve(cli_value, yaml_value, default):
    """CLI overrides YAML; YAML overrides hard-coded default."""
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default


def main(argv):
    args = parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]

    config = load_config(args.config)
    pool_cfg = config.get('detectors', [])
    max_size = max(sizes)
    if len(pool_cfg) < max_size:
        print(f"Error: config pool has {len(pool_cfg)} detectors but max "
              f"ensemble size requested is {max_size}.", file=sys.stderr)
        sys.exit(1)

    detector_criterion = _resolve(
        args.detector_decision_criteria,
        config.get('detector_decision_criteria'), 'majority').lower()
    ensemble_criterion = _resolve(
        args.ensemble_decision_criteria,
        config.get('ensemble_decision_criteria'), 'majority').lower()
    decision_window = _resolve(
        args.decision_window, config.get('decision_window'), 1)
    suppression_window = _resolve(
        args.suppression_window, config.get('suppression_window'),
        args.tolerance)

    print(f"=" * 80)
    print(f"Synthetic Stream Drift Detector Evaluation")
    print(f"  Dataset:                     {args.dataset}")
    print(f"  Config:                      {args.config}")
    print(f"  Ensemble sizes:              {sizes}")
    print(f"  Tolerance:                   {args.tolerance}")
    print(f"  Suppression window:          {suppression_window}")
    print(f"  Decision window:             {decision_window}")
    print(f"  Detector criterion (lvl 1):  {detector_criterion}")
    print(f"  Ensemble criterion (lvl 2):  {ensemble_criterion}")
    print(f"=" * 80)

    # Validate stream has known drift points and instantiate once to inspect.
    dataset_string, probe_stream = parse_expression(args.dataset)
    if not hasattr(probe_stream, "drifts"):
        print(f"Error: dataset '{dataset_string}' does not expose `drifts`. "
              f"This script requires a synthetic stream with known drift points.",
              file=sys.stderr)
        sys.exit(1)
    known_drifts = list(probe_stream.drifts)
    stream_length = getattr(probe_stream, "n_samples", None)
    print(f"\nDataset info: known_drifts={known_drifts}, n_samples={stream_length}\n")

    for n in sizes:
        print(f"\n{'#' * 80}")
        print(f"# Ensemble size N = {n}")
        print(f"{'#' * 80}")

        detectors = []
        names = []
        for k in range(n):
            det, name = instantiate_detector(
                pool_cfg[k],
                seed=args.seed + k,
                recent_samples_size=args.recent_samples_size,
            )
            detectors.append(det)
            names.append(f"[{k:03d}]{name}")

        # Fresh stream per run so each ensemble sees identical data.
        _, stream = parse_expression(args.dataset)

        per_member, ensemble_raw = run_ensemble(
            stream, detectors, names,
            detector_criterion=detector_criterion,
            ensemble_criterion=ensemble_criterion,
            decision_window=decision_window,
        )

        # Apply suppression independently to each source.
        ensemble_dets = apply_suppression(ensemble_raw, suppression_window)
        member_dets = [apply_suppression(m, suppression_window) for m in per_member]

        # ---- Ensemble metrics ----
        e_tp, e_fp, e_fn, e_delay = evaluate_detections(
            ensemble_dets, known_drifts, args.tolerance
        )
        print(f"\nDetected drift points (ensemble, suppressed): {ensemble_dets}")
        print(f"\nEnsemble metrics (N={n}):")
        print("  " + format_metric_row(f"ENSEMBLE(N={n})", e_tp, e_fp, e_fn, e_delay))
        print(f"ENSEMBLE_N={n}: TP={e_tp} FP={e_fp} FN={e_fn} DELAY={e_delay:.4f}"
              if not math.isnan(e_delay) else
              f"ENSEMBLE_N={n}: TP={e_tp} FP={e_fp} FN={e_fn} DELAY=nan")

        # ---- Per-member metrics ----
        print(f"\nPer-member metrics (N={n}, each as if deployed alone):")
        for j in range(n):
            tp, fp, fn, delay = evaluate_detections(
                member_dets[j], known_drifts, args.tolerance
            )
            print("  " + format_metric_row(names[j], tp, fp, fn, delay))
            delay_str = f"{delay:.4f}" if not math.isnan(delay) else "nan"
            print(f"MEMBER_N={n} idx={j} name={names[j]}: "
                  f"TP={tp} FP={fp} FN={fn} DELAY={delay_str}")


if __name__ == '__main__':
    main(sys.argv[1:])
