#!/usr/bin/env python3
"""
Analyze results from the NUMA grid search.

Reads scalability_results.json from each subdirectory of results_numa_gridsearch/,
extracts speedup at each ensemble size (especially K=32), and prints a ranked
comparison table to identify the best NUMA/Slurm/srun configuration.

Usage:
    python further_studies/analyze_numa_gridsearch.py
    python further_studies/analyze_numa_gridsearch.py --results-dir further_studies/results_numa_gridsearch
    python further_studies/analyze_numa_gridsearch.py --export-csv further_studies/numa_gridsearch_summary.csv
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def load_config_results(config_dir):
    """Load scalability_results.json from a config directory."""
    json_path = config_dir / "scalability_results.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def extract_metrics(results, ensemble_sizes):
    """Extract per-K metrics from results list.

    Returns dict: {K: {speedup, throughput, latency, wall_time, contention, worker_work_ratio}}
    """
    metrics = {}

    # Precompute sequential times per K
    seq_times = {}
    for r in results:
        if r.get("mode") == "sequential":
            k = r["n_detectors"]
            seq_times.setdefault(k, []).append(r["elapsed_sec"])

    for k in ensemble_sizes:
        ens_runs = [r for r in results
                    if r.get("mode") == "ensemble" and r["n_detectors"] == k]
        if not ens_runs:
            continue

        times = [r["elapsed_sec"] for r in ens_runs]
        tps = [r["throughput_sps"] for r in ens_runs]
        lats = [r["latency_ms"] for r in ens_runs]

        seq_mean = np.mean(seq_times[k]) if k in seq_times else 0
        ens_mean = np.mean(times)
        speedup = seq_mean / ens_mean if ens_mean > 0 else 0

        m = {
            "speedup": speedup,
            "speedup_std": np.std([seq_mean / t if t > 0 else 0 for t in times]) if len(times) > 1 else 0,
            "throughput_mean": np.mean(tps),
            "throughput_std": np.std(tps),
            "latency_mean": np.mean(lats),
            "latency_std": np.std(lats),
            "wall_time_mean": ens_mean,
            "wall_time_std": np.std(times),
            "n_runs": len(ens_runs),
        }

        # Track stats (if available)
        work_ratios = [r.get("worker_mean_work_ratio", 0) for r in ens_runs]
        main_wait = [r.get("main_wait_ratio", 0) for r in ens_runs]
        contention = [r.get("contention_penalty", 0) for r in ens_runs]
        m["worker_work_ratio"] = np.mean(work_ratios)
        m["main_wait_ratio"] = np.mean(main_wait)
        m["contention_penalty"] = np.mean(contention)

        metrics[k] = m

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Analyze NUMA grid search results")
    ap.add_argument("--results-dir", type=str,
                    default="further_studies/results_numa_gridsearch",
                    help="Root directory containing per-config subdirectories")
    ap.add_argument("--export-csv", type=str, default=None,
                    help="Export summary table to CSV")
    ap.add_argument("--target-k", type=int, default=32,
                    help="Ensemble size to rank by (default 32)")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"Results directory not found: {results_root}")
        sys.exit(1)

    # Discover all config directories
    config_dirs = sorted([d for d in results_root.iterdir()
                          if d.is_dir() and (d / "scalability_results.json").exists()])

    if not config_dirs:
        print(f"No config directories with scalability_results.json found in {results_root}")
        sys.exit(1)

    print(f"Found {len(config_dirs)} configurations in {results_root}")

    # Load all results
    all_metrics = {}
    all_results_raw = {}
    for d in config_dirs:
        name = d.name
        results = load_config_results(d)
        if results is None:
            print(f"  [SKIP] {name}: no results")
            continue

        # Determine ensemble sizes present
        sizes = sorted(set(r["n_detectors"] for r in results
                          if r.get("mode") == "ensemble"))
        metrics = extract_metrics(results, sizes)
        all_metrics[name] = metrics
        all_results_raw[name] = results

    # Determine all ensemble sizes across configs
    all_sizes = sorted(set(k for m in all_metrics.values() for k in m.keys()))

    if not all_sizes:
        print("No ensemble results found in any configuration.")
        sys.exit(1)

    # ---- Print detailed table per K ----
    for k in all_sizes:
        print(f"\n{'='*120}")
        print(f"  Ensemble size K={k}")
        print(f"{'='*120}")

        configs_with_k = [(name, m[k]) for name, m in all_metrics.items() if k in m]
        configs_with_k.sort(key=lambda x: x[1]["speedup"], reverse=True)

        has_stats = any(m.get("contention_penalty", 0) > 0 for _, m in configs_with_k)

        if has_stats:
            print(f"{'Config':<30} {'Speedup':>8} {'Throughput':>14} {'Latency(ms)':>12} "
                  f"{'Wall(s)':>10} {'Contend':>8} {'Wkr%':>7} {'Wait%':>7}")
            print(f"{'-'*30} {'-'*8} {'-'*14} {'-'*12} {'-'*10} {'-'*8} {'-'*7} {'-'*7}")
        else:
            print(f"{'Config':<30} {'Speedup':>8} {'Throughput':>14} {'Latency(ms)':>12} "
                  f"{'Wall(s)':>10}")
            print(f"{'-'*30} {'-'*8} {'-'*14} {'-'*12} {'-'*10}")

        for name, m in configs_with_k:
            if has_stats:
                print(f"{name:<30} {m['speedup']:>7.2f}x "
                      f"{m['throughput_mean']:>8.1f}+/-{m['throughput_std']:>4.1f} "
                      f"{m['latency_mean']:>7.2f}+/-{m['latency_std']:>4.2f} "
                      f"{m['wall_time_mean']:>7.2f} "
                      f"{m['contention_penalty']:>6.2f}x "
                      f"{m['worker_work_ratio']*100:>5.1f}% "
                      f"{m['main_wait_ratio']*100:>5.1f}%")
            else:
                print(f"{name:<30} {m['speedup']:>7.2f}x "
                      f"{m['throughput_mean']:>8.1f}+/-{m['throughput_std']:>4.1f} "
                      f"{m['latency_mean']:>7.2f}+/-{m['latency_std']:>4.2f} "
                      f"{m['wall_time_mean']:>7.2f}")

    # ---- Summary: ranked by target K ----
    target = args.target_k
    print(f"\n{'='*120}")
    print(f"  RANKING by speedup at K={target}")
    print(f"{'='*120}")

    ranked = [(name, m.get(target, {}).get("speedup", 0))
              for name, m in all_metrics.items()]
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<6} {'Config':<30} {'Speedup':>8} {'Throughput':>14} {'Latency(ms)':>12}")
    print(f"{'-'*6} {'-'*30} {'-'*8} {'-'*14} {'-'*12}")
    for rank, (name, speedup) in enumerate(ranked, 1):
        m = all_metrics[name].get(target, {})
        tps = m.get("throughput_mean", 0)
        lat = m.get("latency_mean", 0)
        print(f"{rank:<6} {name:<30} {speedup:>7.2f}x {tps:>10.1f} {lat:>10.2f}")

    # ---- Best configuration details ----
    best_name, best_speedup = ranked[0]
    print(f"\n{'='*120}")
    print(f"  BEST CONFIGURATION: {best_name}")
    print(f"  Speedup at K={target}: {best_speedup:.2f}x")
    print(f"{'='*120}")

    # Print env_info if available
    env_file = results_root / best_name / "env_info.txt"
    if env_file.exists():
        print(f"\n  Environment info ({env_file}):")
        print("  " + "-" * 80)
        for line in env_file.read_text().splitlines():
            print(f"  {line}")

    # ---- Export CSV ----
    if args.export_csv:
        import csv
        with open(args.export_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["config", "K", "speedup", "throughput_mean", "throughput_std",
                      "latency_mean", "latency_std", "wall_time_mean", "wall_time_std",
                      "contention_penalty", "worker_work_ratio", "main_wait_ratio"]
            w.writerow(header)
            for name in sorted(all_metrics.keys()):
                for k in all_sizes:
                    if k not in all_metrics[name]:
                        continue
                    m = all_metrics[name][k]
                    w.writerow([name, k, m["speedup"], m["throughput_mean"],
                                m["throughput_std"], m["latency_mean"], m["latency_std"],
                                m["wall_time_mean"], m["wall_time_std"],
                                m.get("contention_penalty", ""),
                                m.get("worker_work_ratio", ""),
                                m.get("main_wait_ratio", "")])
        print(f"\nCSV exported to: {args.export_csv}")

    # ---- Speedup curve comparison ----
    print(f"\n{'='*120}")
    print(f"  Speedup curve comparison (all configs)")
    print(f"{'='*120}")

    k_header = " ".join(f"K={k:<5}" for k in all_sizes)
    print(f"{'Config':<30} {k_header}")
    print(f"{'-'*30} {'-'*len(k_header)}")
    for name in sorted(all_metrics.keys()):
        vals = []
        for k in all_sizes:
            if k in all_metrics[name]:
                vals.append(f"{all_metrics[name][k]['speedup']:>5.1f}x")
            else:
                vals.append(f"{'--':>6}")
        print(f"{name:<30} {' '.join(vals)}")


if __name__ == "__main__":
    main()
