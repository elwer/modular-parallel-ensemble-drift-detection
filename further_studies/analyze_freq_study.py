#!/usr/bin/env python3
"""
Analyze CPU frequency study results.

Compares speedup across frequency levels (high/mid/low) for each NUMA config.
Tests the hypothesis: if memory-bound, lower CPU frequency → higher speedup.

Usage:
    python further_studies/analyze_freq_study.py
    python further_studies/analyze_freq_study.py --export-csv further_studies/freq_study_summary.csv
    python further_studies/analyze_freq_study.py --target-k 32
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


FREQ_ORDER = ["high", "mid", "low"]
FREQ_MHZ = {"high": 3350, "mid": 2000, "low": 1500}


def load_config_results(config_dir):
    json_path = config_dir / "scalability_results.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def extract_speedup(results, ensemble_sizes):
    """Extract speedup per K from results."""
    speedups = {}
    for k in ensemble_sizes:
        seq_runs = [r for r in results
                    if r.get("mode") == "sequential" and r["n_detectors"] == k]
        par_runs = [r for r in results
                    if r.get("mode") == "ensemble" and r["n_detectors"] == k]
        if not seq_runs or not par_runs:
            continue
        seq_mean = np.mean([r["elapsed_sec"] for r in seq_runs])
        par_mean = np.mean([r["elapsed_sec"] for r in par_runs])
        speedups[k] = seq_mean / par_mean if par_mean > 0 else 0
    return speedups


def extract_throughput(results, ensemble_sizes):
    """Extract parallel throughput per K."""
    throughputs = {}
    for k in ensemble_sizes:
        par_runs = [r for r in results
                    if r.get("mode") == "ensemble" and r["n_detectors"] == k]
        if not par_runs:
            continue
        throughputs[k] = np.mean([r["throughput_sps"] for r in par_runs])
    return throughputs


def extract_wall_time(results, ensemble_sizes):
    """Extract parallel wall time per K."""
    times = {}
    for k in ensemble_sizes:
        par_runs = [r for r in results
                    if r.get("mode") == "ensemble" and r["n_detectors"] == k]
        if not par_runs:
            continue
        times[k] = np.mean([r["elapsed_sec"] for r in par_runs])
    return times


def main():
    ap = argparse.ArgumentParser(description="Analyze CPU frequency study results")
    ap.add_argument("--results-dir", type=str,
                    default="further_studies/results_freq_study",
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

    # Parse config names into (base_config, freq_label)
    configs = {}
    for d in config_dirs:
        name = d.name
        parts = name.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in FREQ_ORDER:
            continue
        base, freq = parts
        if base not in configs:
            configs[base] = {}
        results = load_config_results(d)
        if results is None:
            continue
        ensemble_sizes = sorted(set(r["n_detectors"] for r in results
                                    if r.get("mode") == "ensemble"))
        configs[base][freq] = {
            "speedup": extract_speedup(results, ensemble_sizes),
            "throughput": extract_throughput(results, ensemble_sizes),
            "wall_time": extract_wall_time(results, ensemble_sizes),
            "ensemble_sizes": ensemble_sizes,
        }

    if not configs:
        print("No valid config results found.")
        sys.exit(1)

    # ---- Print speedup comparison table ----
    target_k = args.target_k
    print(f"\n{'='*100}")
    print(f"  CPU Frequency Study — Speedup at K={target_k}")
    print(f"  Hypothesis: if memory-bound, lower freq → higher speedup")
    print(f"  Frequencies: high={FREQ_MHZ['high']} MHz, mid={FREQ_MHZ['mid']} MHz, low={FREQ_MHZ['low']} MHz")
    print(f"{'='*100}")
    print(f"  {'Config':<28} {'High (3350)':>12} {'Mid (2000)':>12} {'Low (1500)':>12} "
          f"{'Δ(H→L)':>10} {'Trend':>8}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    ranked = []
    for base in sorted(configs.keys()):
        freqs = configs[base]
        s_high = freqs.get("high", {}).get("speedup", {}).get(target_k, None)
        s_mid = freqs.get("mid", {}).get("speedup", {}).get(target_k, None)
        s_low = freqs.get("low", {}).get("speedup", {}).get(target_k, None)

        if s_high is None or s_low is None:
            continue

        delta = s_low - s_high
        trend = "↑" if delta > 0.1 else ("↓" if delta < -0.1 else "→")

        ranked.append((base, s_high, s_mid, s_low, delta, trend))

    # Sort by delta (largest improvement at low freq first)
    ranked.sort(key=lambda x: x[4], reverse=True)

    for base, s_high, s_mid, s_low, delta, trend in ranked:
        s_mid_str = f"{s_mid:.2f}x" if s_mid is not None else "N/A"
        print(f"  {base:<28} {s_high:>10.2f}x {s_mid_str:>12} {s_low:>10.2f}x "
              f"{delta:>+8.2f}x {trend:>8}")

    # ---- Print full speedup table across all K ----
    print(f"\n{'='*100}")
    print(f"  Full Speedup Table (all K)")
    print(f"{'='*100}")

    all_ks = set()
    for freqs in configs.values():
        for freq_data in freqs.values():
            all_ks.update(freq_data["ensemble_sizes"])
    all_ks = sorted(all_ks)

    for base in sorted(configs.keys()):
        freqs = configs[base]
        print(f"\n  --- {base} ---")
        header = f"  {'K':>6}"
        for fl in FREQ_ORDER:
            header += f" {fl:>10}"
        header += f" {'Δ(H→L)':>10}"
        print(header)
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for k in all_ks:
            row = f"  {k:>6}"
            vals = {}
            for fl in FREQ_ORDER:
                v = freqs.get(fl, {}).get("speedup", {}).get(k, None)
                vals[fl] = v
                row += f" {v:>9.2f}x" if v is not None else f" {'N/A':>10}"
            if vals.get("high") is not None and vals.get("low") is not None:
                d = vals["low"] - vals["high"]
                row += f" {d:>+8.2f}x"
            else:
                row += f" {'N/A':>10}"
            print(row)

    # ---- Print throughput comparison at target K ----
    print(f"\n{'='*100}")
    print(f"  Throughput at K={target_k} (samples/sec)")
    print(f"{'='*100}")
    print(f"  {'Config':<28} {'High':>12} {'Mid':>12} {'Low':>12} {'Δ(H→L)':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for base in sorted(configs.keys()):
        freqs = configs[base]
        t_high = freqs.get("high", {}).get("throughput", {}).get(target_k, None)
        t_mid = freqs.get("mid", {}).get("throughput", {}).get(target_k, None)
        t_low = freqs.get("low", {}).get("throughput", {}).get(target_k, None)
        if t_high is None or t_low is None:
            continue
        delta = t_low - t_high
        t_mid_str = f"{t_mid:.1f}" if t_mid is not None else "N/A"
        print(f"  {base:<28} {t_high:>10.1f} {t_mid_str:>12} {t_low:>10.1f} {delta:>+10.1f}")

    # ---- Summary ----
    print(f"\n{'='*100}")
    print(f"  Summary at K={target_k}")
    print(f"{'='*100}")
    if ranked:
        best_mem_bound = ranked[0]
        worst_mem_bound = ranked[-1]
        print(f"  Most memory-bound (largest speedup increase at low freq):")
        print(f"    {best_mem_bound[0]}: {best_mem_bound[1]:.2f}x → {best_mem_bound[3]:.2f}x (Δ={best_mem_bound[4]:+.2f}x)")
        print(f"  Least memory-bound (speedup decreases or flat at low freq):")
        print(f"    {worst_mem_bound[0]}: {worst_mem_bound[1]:.2f}x → {worst_mem_bound[3]:.2f}x (Δ={worst_mem_bound[4]:+.2f}x)")

        # Count how many configs show the expected trend
        mem_bound_count = sum(1 for r in ranked if r[4] > 0.1)
        total = len(ranked)
        print(f"\n  Configs with speedup increasing at low freq: {mem_bound_count}/{total}")
        if mem_bound_count > total / 2:
            print(f"  → Majority of configs show memory-bound behavior ✓")
        else:
            print(f"  → Majority of configs do NOT show memory-bound behavior ✗")

    # ---- CSV export ----
    if args.export_csv:
        import csv
        with open(args.export_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "K", "freq_label", "freq_mhz", "speedup",
                        "throughput_sps", "wall_time_sec"])
            for base in sorted(configs.keys()):
                freqs = configs[base]
                for k in all_ks:
                    for fl in FREQ_ORDER:
                        fd = freqs.get(fl)
                        if fd is None:
                            continue
                        s = fd["speedup"].get(k)
                        t = fd["throughput"].get(k)
                        wt = fd["wall_time"].get(k)
                        if s is not None:
                            w.writerow([base, k, fl, FREQ_MHZ[fl], s,
                                        t if t is not None else "",
                                        wt if wt is not None else ""])
        print(f"\n  CSV exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
