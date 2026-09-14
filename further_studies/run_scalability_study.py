#!/usr/bin/env python3
"""
Scalability study: deploy 2,4,8,16,32,64,128 random DDs in parallel
using ThreadsDeployment and measure wall-clock throughput.

For each ensemble size K:
  1. Build K random detectors (sampled from the same param spaces as
     generate_scalability_configs.py).
  2. Deploy them via ThreadsDeployment (one worker thread per detector).
  3. Run a synthetic stream through the ensemble.
  4. Record wall-clock time, throughput (samples/sec), and per-sample latency.

Usage:
    python run_scalability_study.py [--stream-length 2000] [--n-repeats 3] \
        [--seed 42] [--output-dir results_scalability]
"""

import os
import sys
import time
import json
import random
import logging
import argparse
from typing import Dict, List, Callable
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.synthetic_f1_multistream_optimize_optuna import build_stream
from detectors.mopedds.threads_deployment import ThreadsDeployment

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

ENSEMBLE_SIZES = [2, 4, 8, 16, 32, 64, 128]

CANDIDATES = ["CSDDM", "D3", "IBDD", "OCDD", "SPLL", "UDetect"]

CLASS_PATH = {
    "CSDDM":  "detectors.csddm.CSDDM",
    "D3":     "detectors.d3.D3",
    "IBDD":   "detectors.ibdd.IBDD",
    "OCDD":   "detectors.ocdd.OCDD",
    "SPLL":   "detectors.spll.SPLL",
    "UDetect": "detectors.udetect.UDetect",
}

# ============================================================
# Random parameter samplers (mirroring generate_scalability_configs.py)
# ============================================================

def _sample_csddm(rng):
    return {
        "n_samples": rng.randint(50, 500),
        "feature_proportion": rng.uniform(0.1, 1.0),
        "n_clusters": rng.randint(2, 30),
        "confidence": rng.choice([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]),
        "recent_samples_size": rng.randint(50, 5000),
    }

def _sample_d3(rng):
    return {
        "n_reference_samples": rng.randint(50, 5000),
        "recent_samples_proportion": rng.uniform(0.05, 0.5),
        "threshold": rng.uniform(0.1, 0.9),
        "recent_samples_size": rng.randint(50, 5000),
    }

def _sample_ibdd(rng):
    return {
        "n_samples": rng.randint(100, 2000),
        "n_consecutive_deviations": rng.randint(1, 20),
        "n_permutations": rng.randint(100, 1000),
        "update_interval": rng.randint(10, 100),
        "recent_samples_size": rng.randint(50, 5000),
    }

def _sample_ocdd(rng):
    return {
        "n_samples": rng.randint(50, 500),
        "threshold": rng.uniform(0.1, 0.9),
        "recent_samples_size": rng.randint(50, 5000),
    }

def _sample_spll(rng):
    return {
        "n_samples": rng.randint(100, 1000),
        "n_clusters": rng.randint(2, 20),
        "threshold": rng.uniform(0.1, 5.0),
        "recent_samples_size": rng.randint(50, 5000),
    }

def _sample_udetect(rng):
    return {
        "n_windows": rng.randint(5, 30),
        "n_samples": rng.randint(20, 200),
        "disjoint_training_windows": rng.choice([True, False]),
        "recent_samples_size": rng.randint(50, 5000),
    }

SAMPLERS: Dict[str, Callable] = {
    "CSDDM": _sample_csddm,
    "D3": _sample_d3,
    "IBDD": _sample_ibdd,
    "OCDD": _sample_ocdd,
    "SPLL": _sample_spll,
    "UDetect": _sample_udetect,
}

# ============================================================
# Pool construction
# ============================================================

def build_pool(rng, n_total=128):
    """Build a pool of n_total detector names with balanced coverage."""
    pool = []
    while len(pool) < n_total:
        pool.append(rng.choice(CANDIDATES))
    return pool[:n_total]


def materialize_pool(pool_names, rng):
    """Sample params and instantiate detector objects."""
    from main_synthetic import get_detector_class
    detectors = []
    for i, name in enumerate(pool_names):
        params = SAMPLERS[name](rng)
        params["seed"] = rng.randint(0, 99999)
        cls = get_detector_class(CLASS_PATH[name])
        det = cls(**params)
        detectors.append(det)
    return detectors


def build_balanced_pool(n_detectors, rng):
    """All detectors are the same type (OCDD) with identical fast params."""
    names = ["OCDD"] * n_detectors
    params = {"n_samples": 50, "threshold": 0.5, "recent_samples_size": 50}
    from main_synthetic import get_detector_class
    cls = get_detector_class(CLASS_PATH["OCDD"])
    detectors = []
    for i in range(n_detectors):
        det = cls(seed=rng.randint(0, 99999), **params)
        detectors.append(det)
    return names, detectors


def build_unbalanced_pool(n_detectors, rng):
    """One slow detector (D3 with large n_reference_samples) + K-1 fast (OCDD)."""
    from main_synthetic import get_detector_class
    names = ["D3"] + ["OCDD"] * (n_detectors - 1)
    detectors = []
    slow_params = {"n_reference_samples": 5000, "recent_samples_proportion": 0.5,
                    "threshold": 0.5, "recent_samples_size": 5000,
                    "seed": rng.randint(0, 99999)}
    cls_d3 = get_detector_class(CLASS_PATH["D3"])
    detectors.append(cls_d3(**slow_params))
    fast_params = {"n_samples": 50, "threshold": 0.5, "recent_samples_size": 50}
    cls_ocdd = get_detector_class(CLASS_PATH["OCDD"])
    for i in range(n_detectors - 1):
        det = cls_ocdd(seed=rng.randint(0, 99999), **fast_params)
        detectors.append(det)
    return names, detectors

# ============================================================
# Scalability benchmark
# ============================================================

class _DummyMOPEDDS:
    """Minimal stand-in so ThreadsDeployment can read sample_counter/in_suppression."""
    def __init__(self):
        self.sample_counter = 0
        self.in_suppression = False


def run_single_benchmark(detector, stream_length, drift_frequency, seed):
    """Run a single detector sequentially on a stream (no threading)."""
    stream = build_stream("SineClusters", drift_frequency, stream_length, seed)
    stream_iter = iter(stream)
    first_x, _ = next(stream_iter)

    # Warm up
    detector.update(first_x)

    t0 = time.perf_counter()
    n_samples = 1
    drift_count = 0
    for x, _ in stream_iter:
        if detector.update(x):
            drift_count += 1
        n_samples += 1
    elapsed = time.perf_counter() - t0

    return {
        "n_detectors": 1,
        "stream_length": n_samples,
        "elapsed_sec": elapsed,
        "throughput_sps": n_samples / elapsed if elapsed > 0 else 0,
        "latency_ms": (elapsed / n_samples) * 1000 if n_samples > 0 else 0,
        "drift_count": drift_count,
    }


def run_scalability_benchmark(n_detectors, stream_length, drift_frequency,
                              seed, decision_window=10, scenario="random",
                              track_stats=False):
    """Deploy n_detectors DDs via ThreadsDeployment and run a stream."""
    rng = random.Random(seed)

    if scenario == "balanced":
        pool_names, detectors = build_balanced_pool(n_detectors, rng)
    elif scenario == "unbalanced":
        pool_names, detectors = build_unbalanced_pool(n_detectors, rng)
    else:  # random
        pool_names = build_pool(rng, n_total=n_detectors)
        detectors = materialize_pool(pool_names, rng)

    stream = build_stream("SineClusters", drift_frequency, stream_length, seed)

    dummy_mopedds = _DummyMOPEDDS()
    deployment = ThreadsDeployment(
        detectors,
        verbose=False,
        mopedds=dummy_mopedds,
        detector_decision_criteria="majority",
        decision_window=decision_window,
        track_stats=track_stats,
    )
    deployment.initialize()

    # Warm up: process first sample to trigger any lazy init
    stream_iter = iter(stream)
    first_x, _ = next(stream_iter)
    dummy_mopedds.sample_counter = 1
    for slot in deployment.slots:
        slot.data = first_x
        slot.result_ready = False
        slot.sample_id = 1
    # Wait for all workers to process first sample
    pending = set(range(len(deployment.slots)))
    while pending:
        for idx in list(pending):
            if deployment.slots[idx].result_ready:
                pending.remove(idx)

    # Timed run
    t0 = time.perf_counter()
    main_wait_time = 0.0
    n_samples = 1  # already processed first
    drift_count = 0
    for x, _ in stream_iter:
        dummy_mopedds.sample_counter += 1
        sid = dummy_mopedds.sample_counter
        for slot in deployment.slots:
            slot.data = x
            slot.result_ready = False
            slot.sample_id = sid
        # Wait for all results
        pending = set(range(len(deployment.slots)))
        results = [False] * len(deployment.slots)
        _wait_t0 = time.perf_counter() if track_stats else 0.0
        while pending:
            for idx in list(pending):
                if deployment.slots[idx].result_ready:
                    results[idx] = deployment.slots[idx].result
                    pending.remove(idx)
        if track_stats:
            main_wait_time += time.perf_counter() - _wait_t0
        # Majority vote
        if sum(results) >= (len(results) + 1) // 2:
            drift_count += 1
        n_samples += 1
    elapsed = time.perf_counter() - t0

    # Collect worker stats before shutdown
    worker_stats = []
    if track_stats:
        worker_stats = deployment.get_worker_stats()

    deployment.shutdown()

    result = {
        "n_detectors": n_detectors,
        "stream_length": n_samples,
        "elapsed_sec": elapsed,
        "throughput_sps": n_samples / elapsed if elapsed > 0 else 0,
        "latency_ms": (elapsed / n_samples) * 1000 if n_samples > 0 else 0,
        "drift_count": drift_count,
        "scenario": scenario,
    }

    if track_stats:
        result["main_wait_time"] = main_wait_time
        result["main_wait_ratio"] = main_wait_time / elapsed if elapsed > 0 else 0.0
        result["worker_stats"] = worker_stats
        if worker_stats:
            result["worker_mean_work_ratio"] = (
                sum(w["work_ratio"] for w in worker_stats) / len(worker_stats)
            )
            result["worker_min_work_ratio"] = min(w["work_ratio"] for w in worker_stats)
            result["worker_max_work_ratio"] = max(w["work_ratio"] for w in worker_stats)

    return result


def main():
    ap = argparse.ArgumentParser(description="Scalability study: ThreadsDeployment with increasing ensemble sizes")
    ap.add_argument("--stream-length", type=int, default=50000,
                    help="Number of samples per stream (keep large enough for external monitors)")
    ap.add_argument("--drift-frequency", type=int, default=100)
    ap.add_argument("--n-repeats", type=int, default=3,
                    help="Number of repeats per ensemble size (for variance)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="results_scalability")
    ap.add_argument("--max-ensemble-size", type=int, default=None,
                    help="Limit max ensemble size (e.g. 8 for local runs)")
    ap.add_argument("--scenario", type=str, default="random",
                    choices=["random", "balanced", "unbalanced"],
                    help="Scenario: random (mixed DDs), balanced (all same fast DD), "
                         "unbalanced (1 slow DD + K-1 fast DDs)")
    ap.add_argument("--track-stats", action="store_true",
                    help="Track per-worker work vs busy-wait time and main-thread wait time")
    ap.add_argument("--use-jumper", action="store_true",
                    help="Enable JUmPER performance monitoring for each ensemble run")
    ap.add_argument("--jumper-sampling-interval", type=float, default=2.0,
                    help="JUmPER sampling interval in seconds (default 2.0)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ensemble_sizes = ENSEMBLE_SIZES
    if args.max_ensemble_size is not None:
        ensemble_sizes = [s for s in ENSEMBLE_SIZES if s <= args.max_ensemble_size]
    max_pool_size = max(ensemble_sizes) if ensemble_sizes else max(ENSEMBLE_SIZES)

    all_results = []

    # ---- Single-detector baselines ----
    logger.info(f"\n{'='*60}")
    logger.info(f"Single-detector baselines (sequential, no threading)")
    logger.info(f"{'='*60}")

    for rep in range(args.n_repeats):
        seed = args.seed + rep * 1000
        rng = random.Random(seed)
        if args.scenario == "balanced":
            pool_names, detectors = build_balanced_pool(max_pool_size, rng)
        elif args.scenario == "unbalanced":
            pool_names, detectors = build_unbalanced_pool(max_pool_size, rng)
        else:
            pool_names = build_pool(rng, n_total=max_pool_size)
            detectors = materialize_pool(pool_names, rng)
        for i, det in enumerate(detectors):
            logger.info(f"  Rep {rep+1}/{args.n_repeats} detector {i+1}/{len(detectors)} "
                        f"({pool_names[i]})")
            result = run_single_benchmark(
                det, args.stream_length, args.drift_frequency, seed)
            result["rep"] = rep
            result["seed"] = seed
            result["mode"] = "single"
            result["detector_type"] = pool_names[i]
            result["detector_idx"] = i
            all_results.append(result)
            logger.info(f"  -> {result['throughput_sps']:.1f} sps, "
                        f"{result['latency_ms']:.2f} ms/sample")

    # ---- Ensemble runs ----
    for size in ensemble_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Ensemble size: {size} detectors (ThreadsDeployment)")
        logger.info(f"{'='*60}")

        for rep in range(args.n_repeats):
            seed = args.seed + rep * 1000
            logger.info(f"  Rep {rep+1}/{args.n_repeats} (seed={seed})")

            jumper_service = None
            if args.use_jumper:
                from jumper_extension.core.service import build_perfmonitor_service
                jumper_service = build_perfmonitor_service()
                jumper_service.start_monitoring(args.jumper_sampling_interval)
                jumper_result_file = os.path.join(
                    args.output_dir,
                    f"jumper_{args.scenario}_K{size}_rep{rep}.csv")

            try:
                if jumper_service is not None:
                    with jumper_service.monitored():
                        result = run_scalability_benchmark(
                            n_detectors=size,
                            stream_length=args.stream_length,
                            drift_frequency=args.drift_frequency,
                            seed=seed,
                            scenario=args.scenario,
                            track_stats=args.track_stats,
                        )
                else:
                    result = run_scalability_benchmark(
                        n_detectors=size,
                        stream_length=args.stream_length,
                        drift_frequency=args.drift_frequency,
                        seed=seed,
                        scenario=args.scenario,
                        track_stats=args.track_stats,
                    )
            finally:
                if jumper_service is not None:
                    jumper_service.export_perfdata(file=jumper_result_file, level="slurm")
                    jumper_service.stop_monitoring()
                    result["jumper_file"] = jumper_result_file
                    logger.info(f"  JUmPER data: {jumper_result_file}")

            result["rep"] = rep
            result["seed"] = seed
            result["mode"] = "ensemble"
            all_results.append(result)
            logger.info(f"  -> {result['throughput_sps']:.1f} sps, "
                        f"{result['latency_ms']:.2f} ms/sample, "
                        f"{result['elapsed_sec']:.2f}s total")

    # Summary table
    import numpy as np

    # Single-detector baselines
    single_runs = [r for r in all_results if r.get("mode") == "single"]
    single_times = [r["elapsed_sec"] for r in single_runs]
    single_lats = [r["latency_ms"] for r in single_runs]
    single_tps = [r["throughput_sps"] for r in single_runs]
    # Baseline = slowest single detector (the bottleneck in a sequential pipeline)
    slowest_single_time = max(single_times) if single_times else 0
    slowest_single_lat = max(single_lats) if single_lats else 0
    slowest_single_tps = min(single_tps) if single_tps else 0
    # Also report mean and fastest for context
    mean_single_time = np.mean(single_times) if single_times else 0
    mean_single_lat = np.mean(single_lats) if single_lats else 0
    mean_single_tps = np.mean(single_tps) if single_tps else 0
    fastest_single_time = min(single_times) if single_times else 0
    fastest_single_lat = min(single_lats) if single_lats else 0

    print(f"\n{'='*95}")
    print(f"SCALABILITY SUMMARY  (stream_length={args.stream_length}, "
          f"drift_freq={args.drift_frequency}, repeats={args.n_repeats}, "
          f"scenario={args.scenario})")
    print(f"{'='*95}")
    print(f"\n--- Single-detector baselines (sequential, no threading) ---")
    print(f"  Fastest: {fastest_single_time:.3f}s  ({1000/fastest_single_time:.0f} sps)")
    print(f"  Mean:    {mean_single_time:.3f}s  ({mean_single_tps:.0f} sps, {mean_single_lat:.2f} ms/sample)")
    print(f"  Slowest: {slowest_single_time:.3f}s  ({slowest_single_tps:.0f} sps, {slowest_single_lat:.2f} ms/sample)")
    print(f"  (across {len(single_runs)} individual detector runs)")
    print(f"  Baseline = slowest single DD (bottleneck if run sequentially)")

    print(f"\n--- Ensemble (ThreadsDeployment, parallel, scenario={args.scenario}) ---")
    if args.track_stats:
        print(f"{'K':>6} {'Throughput (sps)':>20} {'Latency (ms)':>15} "
              f"{'Wall time (s)':>15} {'Overhead':>10} "
              f"{'Wkr work%':>10} {'Main wait%':>11}")
        print(f"{'-'*6} {'-'*20} {'-'*15} {'-'*15} {'-'*10} {'-'*10} {'-'*11}")
    else:
        print(f"{'K':>6} {'Throughput (sps)':>20} {'Latency (ms)':>15} "
              f"{'Wall time (s)':>15} {'Overhead':>10}")
        print(f"{'-'*6} {'-'*20} {'-'*15} {'-'*15} {'-'*10}")

    for size in ensemble_sizes:
        runs = [r for r in all_results if r.get("mode") == "ensemble" and r["n_detectors"] == size]
        tps = [r["throughput_sps"] for r in runs]
        lats = [r["latency_ms"] for r in runs]
        times = [r["elapsed_sec"] for r in runs]
        mean_ens_time = np.mean(times) if times else 0
        # Overhead = ensemble_wall_time / slowest_single_time
        # Ideal = 1.0 (all K detectors finish in the time of the slowest one)
        overhead = mean_ens_time / slowest_single_time if slowest_single_time > 0 else 0
        if args.track_stats:
            wkr_ratios = [r.get("worker_mean_work_ratio", 0) for r in runs]
            main_ratios = [r.get("main_wait_ratio", 0) for r in runs]
            print(f"{size:>6} {np.mean(tps):>10.1f}+/-{np.std(tps):>5.1f} "
                  f"{np.mean(lats):>8.2f}+/-{np.std(lats):>4.2f} "
                  f"{np.mean(times):>8.2f}+/-{np.std(times):>4.2f} "
                  f"{overhead:>6.2f}x "
                  f"{np.mean(wkr_ratios)*100:>8.1f}% "
                  f"{np.mean(main_ratios)*100:>9.1f}%")
        else:
            print(f"{size:>6} {np.mean(tps):>10.1f}+/-{np.std(tps):>5.1f} "
                  f"{np.mean(lats):>8.2f}+/-{np.std(lats):>4.2f} "
                  f"{np.mean(times):>8.2f}+/-{np.std(times):>4.2f} "
                  f"{overhead:>6.2f}x")

    print(f"\n  Overhead = ensemble_wall_time / slowest_single_time")
    print(f"  Ideal overhead = 1.0x (K detectors in parallel take same time as slowest alone)")
    print(f"  Overhead > 1.0 = communication/synchronization cost of threading")
    if args.track_stats:
        print(f"  Wkr work%  = mean fraction of wall time workers spend in detector.update()")
        print(f"  Main wait% = fraction of wall time main thread spends spinning on results")

    # Save JSON
    out_path = os.path.join(args.output_dir, "scalability_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Save CSV
    import csv
    csv_path = os.path.join(args.output_dir, "scalability_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["mode", "n_detectors", "rep", "throughput_sps", "latency_ms",
                  "elapsed_sec", "drift_count", "detector_type", "scenario"]
        if args.track_stats:
            header += ["main_wait_time", "main_wait_ratio",
                       "worker_mean_work_ratio", "worker_min_work_ratio",
                       "worker_max_work_ratio"]
        w.writerow(header)
        for r in all_results:
            row = [r.get("mode", "ensemble"), r["n_detectors"], r["rep"],
                   r["throughput_sps"], r["latency_ms"],
                   r["elapsed_sec"], r["drift_count"],
                   r.get("detector_type", ""), r.get("scenario", "")]
            if args.track_stats:
                row += [r.get("main_wait_time", ""), r.get("main_wait_ratio", ""),
                        r.get("worker_mean_work_ratio", ""),
                        r.get("worker_min_work_ratio", ""),
                        r.get("worker_max_work_ratio", "")]
            w.writerow(row)
    logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
