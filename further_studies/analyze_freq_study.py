#!/usr/bin/env python3
"""
Analyze CPU frequency study results.

Compares speedup across frequency levels for each NUMA config.
Tests the hypothesis: if memory-bound, lower CPU frequency -> higher speedup.

Usage:
    python further_studies/analyze_freq_study.py
    python further_studies/analyze_freq_study.py --results-dir further_studies/results_freq_study_intel
    python further_studies/analyze_freq_study.py --export-csv further_studies/freq_study_summary.csv
    python further_studies/analyze_freq_study.py --target-k 32
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Known frequency labels and their MHz values.
# New labels can be added here; unknown labels are auto-detected from dir names.
KNOWN_FREQ_MHZ = {
    "high": 3350,
    "mid": 2000,
    "low": 1500,
    "ultra_low": 800,
}


def load_config_results(config_dir):
    json_path = config_dir / "scalability_results.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def detect_freq_mhz(config_dir, freq_label):
    """Try to read actual frequency from env_info.txt, fall back to KNOWN_FREQ_MHZ."""
    env_info = config_dir / "env_info.txt"
    if env_info.exists():
        try:
            with open(env_info) as f:
                for line in f:
                    if "scaling_setspeed" in line.lower() and "khz" in line.lower():
                        parts = line.split()
                        for p in parts:
                            p = p.strip()
                            if p.isdigit() and len(p) >= 5:
                                return int(p) // 1000
        except Exception:
            pass
    return KNOWN_FREQ_MHZ.get(freq_label, 0)


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
    # Auto-detect frequency labels from directory names
    configs = {}
    detected_freq_labels = set()
    for d in config_dirs:
        name = d.name
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        base, freq = parts
        detected_freq_labels.add(freq)
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
            "freq_mhz": detect_freq_mhz(d, freq),
        }

    if not configs:
        print("No valid config results found.")
        sys.exit(1)

    # Build ordered frequency list (sorted by MHz descending)
    freq_order = sorted(detected_freq_labels,
                        key=lambda fl: configs[next(iter(configs))].get(fl, {}).get("freq_mhz", 0),
                        reverse=True)

    # Map freq labels to MHz for display
    freq_mhz = {}
    for fl in freq_order:
        for base in configs:
            if fl in configs[base]:
                freq_mhz[fl] = configs[base][fl].get("freq_mhz", KNOWN_FREQ_MHZ.get(fl, 0))
                break
        if fl not in freq_mhz:
            freq_mhz[fl] = KNOWN_FREQ_MHZ.get(fl, 0)

    # Highest and lowest freq labels for delta calculation
    highest_fl = freq_order[0] if freq_order else None
    lowest_fl = freq_order[-1] if len(freq_order) > 1 else highest_fl

    # ---- Print speedup comparison table ----
    target_k = args.target_k
    print(f"\n{'='*100}")
    print(f"  CPU Frequency Study - Speedup at K={target_k}")
    print(f"  Hypothesis: if memory-bound, lower freq -> higher speedup")
    freq_desc = ", ".join(f"{fl}={freq_mhz[fl]} MHz" for fl in freq_order)
    print(f"  Frequencies: {freq_desc}")
    print(f"{'='*100}")

    # Build header
    header = f"  {'Config':<28}"
    for fl in freq_order:
        header += f" {fl}({freq_mhz[fl]}):>12}"
    if highest_fl and lowest_fl and highest_fl != lowest_fl:
        header += f" {'D':>10} {'Trend':>8}"
    print(header)
    sep = f"  {'-'*28}"
    for _ in freq_order:
        sep += f" {'-'*12}"
    if highest_fl and lowest_fl and highest_fl != lowest_fl:
        sep += f" {'-'*10} {'-'*8}"
    print(sep)

    ranked = []
    for base in sorted(configs.keys()):
        freqs = configs[base]
        vals = {}
        for fl in freq_order:
            vals[fl] = freqs.get(fl, {}).get("speedup", {}).get(target_k, None)

        if vals.get(highest_fl) is None or vals.get(lowest_fl) is None:
            continue

        delta = vals[lowest_fl] - vals[highest_fl]
        trend = "↑" if delta > 0.1 else ("↓" if delta < -0.1 else "→")

        ranked.append((base, vals, delta, trend))

    # Sort by delta (largest improvement at low freq first)
    ranked.sort(key=lambda x: x[2], reverse=True)

    for base, vals, delta, trend in ranked:
        row = f"  {base:<28}"
        for fl in freq_order:
            v = vals.get(fl)
            row += f" {v:>10.2f}x" if v is not None else f" {'N/A':>12}"
        if highest_fl != lowest_fl:
            row += f" {delta:>+8.2f}x {trend:>8}"
        print(row)

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
        for fl in freq_order:
            header += f" {fl:>10}"
        if highest_fl != lowest_fl:
            header += f" {'D':>10}"
        print(header)
        sep = f"  {'-'*6}"
        for _ in freq_order:
            sep += f" {'-'*10}"
        if highest_fl != lowest_fl:
            sep += f" {'-'*10}"
        print(sep)
        for k in all_ks:
            row = f"  {k:>6}"
            vals = {}
            for fl in freq_order:
                v = freqs.get(fl, {}).get("speedup", {}).get(k, None)
                vals[fl] = v
                row += f" {v:>9.2f}x" if v is not None else f" {'N/A':>10}"
            if highest_fl != lowest_fl and vals.get(highest_fl) is not None and vals.get(lowest_fl) is not None:
                d = vals[lowest_fl] - vals[highest_fl]
                row += f" {d:>+8.2f}x"
            elif highest_fl != lowest_fl:
                row += f" {'N/A':>10}"
            print(row)

    # ---- Print throughput comparison at target K ----
    print(f"\n{'='*100}")
    print(f"  Throughput at K={target_k} (samples/sec)")
    print(f"{'='*100}")
    header = f"  {'Config':<28}"
    for fl in freq_order:
        header += f" {fl:>12}"
    if highest_fl != lowest_fl:
        header += f" {'D':>12}"
    print(header)
    sep = f"  {'-'*28}"
    for _ in freq_order:
        sep += f" {'-'*12}"
    if highest_fl != lowest_fl:
        sep += f" {'-'*12}"
    print(sep)
    for base in sorted(configs.keys()):
        freqs = configs[base]
        row = f"  {base:<28}"
        vals = {}
        for fl in freq_order:
            v = freqs.get(fl, {}).get("throughput", {}).get(target_k, None)
            vals[fl] = v
            row += f" {v:>10.1f}" if v is not None else f" {'N/A':>12}"
        if highest_fl != lowest_fl and vals.get(highest_fl) is not None and vals.get(lowest_fl) is not None:
            d = vals[lowest_fl] - vals[highest_fl]
            row += f" {d:>+10.1f}"
        elif highest_fl != lowest_fl:
            row += f" {'N/A':>12}"
        print(row)

    # ---- Summary ----
    print(f"\n{'='*100}")
    print(f"  Summary at K={target_k}")
    print(f"{'='*100}")
    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        print(f"  Most synchronization-bound (largest speedup increase at low freq):")
        print(f"    {best[0]}: {best[1].get(highest_fl, 0):.2f}x -> {best[1].get(lowest_fl, 0):.2f}x (D={best[2]:+.2f}x)")
        print(f"  Least synchronization-bound (speedup decreases or flat at low freq):")
        print(f"    {worst[0]}: {worst[1].get(highest_fl, 0):.2f}x -> {worst[1].get(lowest_fl, 0):.2f}x (D={worst[2]:+.2f}x)")

        sync_bound_count = sum(1 for r in ranked if r[2] > 0.1)
        total = len(ranked)
        print(f"\n  Configs with speedup increasing at low freq: {sync_bound_count}/{total}")
        if sync_bound_count > total / 2:
            print(f"  -> Majority of configs show synchronization-bound behavior")
        else:
            print(f"  -> Majority of configs do NOT show synchronization-bound behavior")

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
                    for fl in freq_order:
                        fd = freqs.get(fl)
                        if fd is None:
                            continue
                        s = fd["speedup"].get(k)
                        t = fd["throughput"].get(k)
                        wt = fd["wall_time"].get(k)
                        if s is not None:
                            w.writerow([base, k, fl, freq_mhz.get(fl, 0), s,
                                        t if t is not None else "",
                                        wt if wt is not None else ""])
        print(f"\n  CSV exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
