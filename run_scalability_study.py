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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# ============================================================
# Scalability benchmark
# ============================================================

class _DummyMOPEDDS:
    """Minimal stand-in so ThreadsDeployment can read sample_counter/in_suppression."""
    def __init__(self):
        self.sample_counter = 0
        self.in_suppression = False


def run_scalability_benchmark(n_detectors, stream_length, drift_frequency,
                              seed, decision_window=10):
    """Deploy n_detectors random DDs via ThreadsDeployment and run a stream."""
    rng = random.Random(seed)
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
        while pending:
            for idx in list(pending):
                if deployment.slots[idx].result_ready:
                    results[idx] = deployment.slots[idx].result
                    pending.remove(idx)
        # Majority vote
        if sum(results) >= (len(results) + 1) // 2:
            drift_count += 1
        n_samples += 1
    elapsed = time.perf_counter() - t0

    deployment.shutdown()

    return {
        "n_detectors": n_detectors,
        "stream_length": n_samples,
        "elapsed_sec": elapsed,
        "throughput_sps": n_samples / elapsed if elapsed > 0 else 0,
        "latency_ms": (elapsed / n_samples) * 1000 if n_samples > 0 else 0,
        "drift_count": drift_count,
    }


def main():
    ap = argparse.ArgumentParser(description="Scalability study: ThreadsDeployment with increasing ensemble sizes")
    ap.add_argument("--stream-length", type=int, default=2000)
    ap.add_argument("--drift-frequency", type=int, default=500)
    ap.add_argument("--n-repeats", type=int, default=3,
                    help="Number of repeats per ensemble size (for variance)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="results_scalability")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for size in ENSEMBLE_SIZES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Ensemble size: {size} detectors")
        logger.info(f"{'='*60}")

        for rep in range(args.n_repeats):
            seed = args.seed + rep * 1000
            logger.info(f"  Rep {rep+1}/{args.n_repeats} (seed={seed})")
            result = run_scalability_benchmark(
                n_detectors=size,
                stream_length=args.stream_length,
                drift_frequency=args.drift_frequency,
                seed=seed,
            )
            result["rep"] = rep
            result["seed"] = seed
            all_results.append(result)
            logger.info(f"  -> {result['throughput_sps']:.1f} sps, "
                        f"{result['latency_ms']:.2f} ms/sample, "
                        f"{result['elapsed_sec']:.2f}s total")

    # Summary table
    print(f"\n{'='*80}")
    print(f"SCALABILITY SUMMARY  (stream_length={args.stream_length}, "
          f"drift_freq={args.drift_frequency}, repeats={args.n_repeats})")
    print(f"{'='*80}")
    print(f"{'K':>6} {'Throughput (sps)':>20} {'Latency (ms)':>15} {'Wall time (s)':>15}")
    print(f"{'-'*6} {'-'*20} {'-'*15} {'-'*15}")

    import numpy as np
    for size in ENSEMBLE_SIZES:
        runs = [r for r in all_results if r["n_detectors"] == size]
        tps = [r["throughput_sps"] for r in runs]
        lats = [r["latency_ms"] for r in runs]
        times = [r["elapsed_sec"] for r in runs]
        print(f"{size:>6} {np.mean(tps):>10.1f}+/-{np.std(tps):>5.1f} "
              f"{np.mean(lats):>8.2f}+/-{np.std(lats):>4.2f} "
              f"{np.mean(times):>8.2f}+/-{np.std(times):>4.2f}")

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
        w.writerow(["n_detectors", "rep", "throughput_sps", "latency_ms", "elapsed_sec", "drift_count"])
        for r in all_results:
            w.writerow([r["n_detectors"], r["rep"], r["throughput_sps"],
                        r["latency_ms"], r["elapsed_sec"], r["drift_count"]])
    logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
